#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creó .env a partir de .env.example: completá GOOGLE_API_KEY y volvé a correr ./run.sh"
  exit 1
fi

docker compose up --build
