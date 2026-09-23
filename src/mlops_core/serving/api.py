"""Prediction API: one route set per model of the domain.

Online and batch inference must agree, so both build features with the domain's
`enrich` - the same function, fed the same context tables. The API never computes a
feature of its own, and it never knows what an item is: each model's request body is the
one the domain declares, and the response names the target instead of assuming one.

Routes are explicit per model (`/models/<name>/predict`), not a path parameter: each
model has its own request body, which FastAPI validates and documents only when the
route knows its type.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import polars as pl
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from mlops_core.adapter import DomainAdapter, ItemRequest, load_adapter
from mlops_core.config import Settings
from mlops_core.ml.registry import ServedModel, load_champion
from mlops_core.storage import read_table

logger = logging.getLogger(__name__)


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    target: str  # what was predicted, named as the domain names it
    prediction: float
    model_version: str
    model_source: str
    # Every feature the request did not supply: what the domain looked up for it, so a
    # surprising prediction can be explained.
    context: dict[str, float | str | None]


class ModelStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    registered_model: str
    model_version: str
    model_source: str
    context_rows: int


class ReloadResult(BaseModel):
    loaded: dict[str, ModelStatus]
    # A model that failed keeps serving what it had, if anything; this says why.
    failed: dict[str, str]


class Service:
    """Holds what is expensive to load: each model's champion and its context tables."""

    def __init__(self, adapter: DomainAdapter, settings: Settings) -> None:
        self.adapter = adapter
        self.config = adapter.config
        self.data_dir = settings.data_dir / self.config.name
        self.tracking_uri = settings.mlflow_tracking_uri
        self._cache_dir = settings.model_cache_dir
        self.served: dict[str, ServedModel] = {}
        self.contexts: dict[str, dict[str, pl.DataFrame]] = {}

    def reload(self, name: str) -> None:
        """Load one model's champion and its context together, or neither.

        Both are read before either is kept: a model whose context failed to load would
        report itself healthy on /health and fail every predict. That happened, when a
        mounted data path was wrong.
        """
        model = self.config.model_named(name)
        served = load_champion(model.training.registered_model, self.tracking_uri, self.cache_dir)
        context = {
            table: read_table(self.data_dir / "clean" / table)
            for table in self.adapter.context_tables(name)
        }
        self.served[name], self.contexts[name] = served, context

    def reload_all(self) -> dict[str, str]:
        """Reload every model; one that fails does not stop the others. Returns why each
        failure failed."""
        failed = {}
        for model in self.config.models:
            try:
                self.reload(model.name)
            except Exception as missing:  # not trained yet, registry down, context gone
                logger.error("%s: no model loaded (%s)", model.name, missing)
                failed[model.name] = str(missing)
        return failed

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir or self.data_dir / "model_cache"

    def ready(self, name: str) -> ServedModel:
        if name not in self.served:
            raise HTTPException(
                503, f"No {name} model loaded: train one or start MLflow, then POST /reload"
            )
        return self.served[name]

    def status(self, name: str) -> ModelStatus:
        served = self.ready(name)
        return ModelStatus(
            registered_model=self.config.model_named(name).training.registered_model,
            model_version=served.version,
            model_source=served.source,
            context_rows=sum(table.height for table in self.contexts[name].values()),
        )

    def predict(self, name: str, request: ItemRequest) -> Prediction:
        served = self.ready(name)
        spec = self.config.model_named(name).spec
        item = request.to_item()
        features = self.adapter.enrich(name, pl.DataFrame([item]), self.contexts[name])
        features = features.with_columns(pl.col(c).cast(pl.Float64) for c in spec.numeric)
        # Built from rows rather than polars.to_pandas(), which needs pyarrow: 156 MB in
        # the image for one conversion. The casts keep the dtypes the model trained on,
        # since pandas cannot infer a column's type from a single missing value: a
        # request that states no producer would arrive as `object` against the batch
        # path's string, which is how online and batch start to drift.
        dtypes = {column: "float64" for column in spec.numeric}
        dtypes |= {column: "str" for column in spec.categorical}
        model_input = pd.DataFrame(features.select(spec.features).to_dicts()).astype(dtypes)
        prediction = served.model.predict(model_input)
        looked_up = [c for c in spec.features if c not in item]
        return Prediction(
            target=spec.target,
            prediction=float(prediction[0]),
            model_version=served.version,
            model_source=served.source,
            context={c: features[c].item() for c in looked_up},
        )


def create_app(adapter: DomainAdapter, settings: Settings) -> FastAPI:
    service = Service(adapter, settings)
    config = adapter.config

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service.reload_all()  # start anyway: /health reports what is missing
        yield

    targets = ", ".join(f"{m.name} ({m.spec.target})" for m in config.models)
    app = FastAPI(
        title=f"{config.name} predictions",
        summary=f"One model per question: {targets}.",
        lifespan=lifespan,
    )

    def get_service() -> Service:
        return service

    Injected = Annotated[Service, Depends(get_service)]

    @app.get("/health")
    def health(service: Injected) -> dict[str, Any]:
        versions = {
            m.name: service.served[m.name].version if m.name in service.served else None
            for m in config.models
        }
        loaded = [version for version in versions.values() if version is not None]
        status = "ok" if len(loaded) == len(versions) else "partial" if loaded else "no model"
        return {"status": status, "models": versions}

    @app.post("/reload", response_model=ReloadResult)
    def reload(service: Injected) -> ReloadResult:
        """Pick up newly promoted champions without restarting the service."""
        failed = service.reload_all()
        loaded = {m.name: service.status(m.name) for m in config.models if m.name not in failed}
        return ReloadResult(loaded=loaded, failed=failed)

    for model in config.models:
        _model_routes(app, model.name, adapter.request_model(model.name), get_service)
    return app


def _model_routes(
    app: FastAPI, name: str, body: type[BaseModel], get_service: Callable[[], Service]
) -> None:
    """`GET /models/<name>` and `POST /models/<name>/predict`, typed with its body."""
    Injected = Annotated[Service, Depends(get_service)]

    @app.get(f"/models/{name}", response_model=ModelStatus, name=f"{name}_status")
    def status(service: Injected) -> ModelStatus:
        return service.status(name)

    # The body's type is the domain's, known only at runtime: FastAPI reads it from the
    # annotation to validate and document the request, which a static checker cannot follow.
    @app.post(f"/models/{name}/predict", response_model=Prediction, name=f"{name}_predict")
    def predict(request: body, service: Injected) -> Prediction:  # type: ignore[valid-type]
        return service.predict(name, request)


settings = Settings()
app = create_app(load_adapter(settings.domain), settings)
