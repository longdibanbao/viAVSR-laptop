#!/bin/bash

docker compose up --build -d

sleep 3

xdg-open http://localhost:8501

docker compose logs -f
