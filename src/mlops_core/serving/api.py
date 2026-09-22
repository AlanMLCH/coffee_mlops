"""Prediction API.

Online and batch inference must agree, so both build features with the domain's
`enrich` - the same function, fed the same context tables. The API never computes a
feature of its own, and it never knows what an item is: the request body is the model
the domain declares, and the response names the target instead of assuming one.
"""

import logging
from collections.abc import AsyncIterator
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


class Service:
    """Holds what is expensive to load: the champion model and the context tables."""

    def __init__(self, adapter: DomainAdapter, settings: Settings) -> None:
        self.adapter = adapter
        self.config = adapter.config
        self.data_dir = settings.data_dir / self.config.name
        self.tracking_uri = settings.mlflow_tracking_uri
        self._cache_dir = settings.model_cache_dir
        self.model: ServedModel | None = None
        self.context: dict[str, pl.DataFrame] = {}

    def reload(self) -> None:
        """Load the champion and its context together, or neither.

        Both are read before either is kept: a model whose context failed to load would
        report itself healthy on /health and fail every /predict. That happened, when a
        mounted data path was wrong.
        """
        model = load_champion(
            self.config.training.registered_model, self.tracking_uri, self.cache_dir
        )
        context = {
            table: read_table(self.data_dir / "clean" / table)
            for table in self.adapter.context_tables()
        }
        self.model, self.context = model, context

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir or self.data_dir / "model_cache"

    def ready(self) -> ServedModel:
        if self.model is None:
            raise HTTPException(
                503, "No model loaded: train one or start MLflow, then POST /reload"
            )
        return self.model

    def predict(self, request: ItemRequest) -> Prediction:
        served = self.ready()
        spec = self.config.model
        item = request.to_item()
        features = self.adapter.enrich(pl.DataFrame([item]), self.context).with_columns(
            pl.col(c).cast(pl.Float64) for c in spec.numeric
        )
        # Built from rows rather than polars.to_pandas(), which needs pyarrow: 156 MB in
        # the image for one conversion. The casts keep the dtypes the model trained on,
        # since pandas cannot infer a numeric column from a single missing value.
        model_input = pd.DataFrame(features.select(spec.features).to_dicts()).astype(
            {column: "float64" for column in spec.numeric}
        )
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
    request_model = adapter.request_model()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            service.reload()
        except Exception as missing:  # start anyway: /health reports the problem
            logger.error("Starting without a model: %s", missing)
        yield

    app = FastAPI(
        title=f"{config.name} predictions",
        summary=f"Predicts {config.model.target} for a single {config.items.noun}.",
        lifespan=lifespan,
    )

    def get_service() -> Service:
        return service

    Injected = Annotated[Service, Depends(get_service)]

    @app.get("/health")
    def health(service: Injected) -> dict[str, Any]:
        served = service.model
        return {
            "status": "ok" if served else "no model",
            "model_version": served.version if served else None,
        }

    @app.get("/model", response_model=ModelStatus)
    def model_status(service: Injected) -> ModelStatus:
        served = service.ready()
        return ModelStatus(
            registered_model=service.config.training.registered_model,
            model_version=served.version,
            model_source=served.source,
            context_rows=sum(table.height for table in service.context.values()),
        )

    @app.post("/reload", response_model=ModelStatus)
    def reload(service: Injected) -> ModelStatus:
        """Pick up a newly promoted champion without restarting the service."""
        service.reload()
        return model_status(service)

    # The body's type is the domain's, known only at runtime: FastAPI reads it from the
    # annotation to validate and document the request, which a static checker cannot follow.
    @app.post("/predict", response_model=Prediction)
    def predict(request: request_model, service: Injected) -> Prediction:  # type: ignore[valid-type]
        return service.predict(request)

    return app


settings = Settings()
app = create_app(load_adapter(settings.domain), settings)
