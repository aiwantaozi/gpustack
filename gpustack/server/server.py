import asyncio
import atexit
from multiprocessing import Process
import os
import secrets
from typing import List
from sqlmodel import Session
import uvicorn
import logging

from gpustack.logging import setup_logging
from gpustack.schemas.models import Model
from gpustack.schemas.users import User, UserCreate
from gpustack.security import get_password_hash
from gpustack.server.app import app
from gpustack.server.config import ServerConfig
from gpustack.server.controller import ModelController
from gpustack.server.db import init_db, get_engine
from gpustack.server.scheduler import Scheduler

logger = logging.getLogger(__name__)


class Server:

    def __init__(self, config: ServerConfig, sub_processes: List[Process] = None):
        if sub_processes is None:
            sub_processes = []
        self._config: ServerConfig = config
        self._sub_processes = sub_processes

        atexit.register(self.at_exit)

    @property
    def all_processes(self):
        return self._sub_processes

    @property
    def config(self):
        return self._config

    async def start(self):
        logger.info("Starting GPUStack server.")

        self._prepare_data()
        self._start_sub_processes()
        self._start_scheduler()
        self._start_controllers()

        # Start FastAPI server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=80,
            access_log=False,
            log_level="error",
        )

        setup_logging()

        logger.info(f"Serving on {config.host}:{config.port}.")
        server = uvicorn.Server(config)
        await server.serve()

    def _prepare_data(self):
        self._setup_data_dir(self._config.data_dir)

        init_db(self._config.database_url)

        engine = get_engine()
        with Session(engine) as session:
            self._init_data(session)

        logger.debug("Data initialization completed.")

    def _start_scheduler(self):
        scheduler = Scheduler()
        asyncio.create_task(scheduler.start())

        logger.debug("Scheduler started.")

    def _start_controllers(self):
        controller = ModelController()
        asyncio.create_task(controller.start())

        logger.debug("Controller started.")

    def _start_sub_processes(self):
        for process in self._sub_processes:
            process.start()

    @staticmethod
    def _setup_data_dir(data_dir: str):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _init_data(self, session: Session):
        init_data_funcs = [self._init_model, self._init_user]
        for init_data_func in init_data_funcs:
            init_data_func(session)

    def _init_model(self, session: Session):
        if self._config.model:
            huggingface_model_id = self._config.model
        else:
            return

        model_name = huggingface_model_id.split("/")[-1]

        model = Model.first_by_field(session=session, field="name", value=model_name)
        if not model:
            model = Model(
                name=model_name,
                source="huggingface",
                huggingface_model_id=huggingface_model_id,
            )
            model.save(session)

            logger.info("Created model: %s", model_name)

    def _init_user(self, session: Session):
        user = User.first_by_field(session=session, field="name", value="admin")
        if not user:
            bootstrap_password = self._config.bootstrap_password
            if not bootstrap_password:
                bootstrap_password = secrets.token_urlsafe(16)
                logger.info("!!!Bootstrap password!!!: %s", bootstrap_password)

            user_create = UserCreate(
                name="admin",
                full_name="System Admin",
                password=bootstrap_password,
                is_admin=True,
            )
            user = User.model_validate(
                user_create,
                update={"hashed_password": get_password_hash(user_create.password)},
            )
            user.save(session)

    def at_exit(self):
        logger.info("Stopping GPUStack server.")
        for process in self._sub_processes:
            process.terminate()
        logger.info("Stopped all processes.")
