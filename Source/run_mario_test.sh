#!/usr/bin/env bash

mario_server () {
    source bin/activate
    python mario_env_server.py
}

mario_client () {
    sleep 1
    python3 mario_env_client.py
}

mario_server &
mario_client &
wait