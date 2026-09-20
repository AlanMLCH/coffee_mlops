"""Prediction API.

Online and batch inference must agree, so both build features with the same
`add_market_context`: a lot graded in year Y sees market year Y-1. The API never
computes features of its own.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import polars as pl
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from coffee_mlops.config import DomainConfig, Settings, load_domain_config
from coffee_mlops.ml.features import add_market_context
from coffee_mlops.ml.registry import ServedModel, load_champion
from coffee_mlops.storage import read_table

logger = logging.getLogger(__name__)

CONTEXT_TABLE = "market_context"


class Lot(BaseModel):
    """What a caller knows about a green coffee lot before it is cupped."""

    country: str
    variety: str | None = None
    processing_method: str | None = None
    color: str | None = None
    altitude_m: float | None = Field(None, ge=0, le=9000)
    moisture_pct: float | None = Field(None, ge=0, le=100)
    category_one_defects: int = Field(0, ge=0)
    category_two_defects: int = Field(0, ge=0)
    quakers: int | None = Field(None, ge=0)
    graded_on: date | None = Field(None, description="Defaults to today (UTC).")


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    total_cup_points: float
    model_version: str
    model_source: str
    # The context the model actually saw, so a surprising prediction can be explained.
    market_context: dict[str, float | None]


class ModelStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    registered_model: str
    model_version: str
    model_source: str
    context_rows: int


class Service:
    """Holds what is expensive to load: the champion model and the market context."""

    def __init__(self, config: DomainConfig, settings: Settings) -> None:
        self.config = config
        self.data_dir = settings.data_dir / config.name
        self.tracking_uri = settings.mlflow_tracking_uri
        self._cache_dir = settings.model_cache_dir
        self.model: ServedModel | None = None
        self.context = pl.DataFrame()

    def reload(self) -> None:
        self.model = load_champion(
            self.config.training.registered_model, self.tracking_uri, self.cache_dir
        )
        self.context = read_table(self.data_dir / "clean" / CONTEXT_TABLE)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir or self.data_dir / "model_cache"

    def ready(self) -> ServedModel:
        if self.model is None:
            raise HTTPException(
                503, "No model loaded: train one or start MLflow, then POST /reload"
            )
        return self.model

    def predict(self, lot: Lot) -> Prediction:
        served = self.ready()
        spec = self.config.model
        item = lot.model_dump(exclude={"graded_on"}) | {
            "grading_date": lot.graded_on or datetime.now(UTC).date()
        }
        features = add_market_context(pl.DataFrame([item]), self.context).with_columns(
            pl.col(c).cast(pl.Float64) for c in spec.numeric
        )
        # Built from rows rather than polars.to_pandas(), which needs pyarrow: 156 MB in
        # the image for one conversion. The casts keep the dtypes the model trained on,
        # since pandas cannot infer a numeric column from a single missing value.
        model_input = pd.DataFrame(features.select(spec.features).to_dicts()).astype(
            {column: "float64" for column in spec.numeric}
        )
        prediction = served.model.predict(model_input)
        context = {c: features[c].item() for c in spec.numeric if c.startswith("ctx_")}
        return Prediction(
            total_cup_points=float(prediction[0]),
            model_version=served.version,
            model_source=served.source,
            market_context=context,
        )


def create_app(config: DomainConfig, settings: Settings) -> FastAPI:
    service = Service(config, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            service.reload()
        except Exception as missing:  # start anyway: /health reports the problem
            logger.error("Starting without a model: %s", missing)
        yield

    app = FastAPI(
        title=f"{config.name} predictions",
        summary=f"Predicts {config.model.target} for a single lot.",
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
            context_rows=service.context.height,
        )

    @app.post("/reload", response_model=ModelStatus)
    def reload(service: Injected) -> ModelStatus:
        """Pick up a newly promoted champion without restarting the service."""
        service.reload()
        return model_status(service)

    @app.post("/predict", response_model=Prediction)
    def predict(lot: Lot, service: Injected) -> Prediction:
        return service.predict(lot)

    return app


settings = Settings()
app = create_app(load_domain_config(settings.domain), settings)
