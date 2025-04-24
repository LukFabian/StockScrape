fastapi dev ../app/main.py
curl http://localhost:8000/api/openapi.json -o ../app/api/openapi.json
openapi-generator-cli generate -i ../app/api/openapi.json -g typescript-axios -o ./generated/
