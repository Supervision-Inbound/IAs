# IAs — Inferencia de llamadas, TMO y alertas clima

## Salidas
- `public/prediccion_diaria.json` — 120 días: { fecha, llamadas_diarias, tmo_diario }
- `public/prediccion_horaria.json` — 120d x 24h: { ts, llamadas_hora, tmo_hora, agentes_requeridos }
- `public/alertas_clima.json` — Rangos por comuna con incremento adicional esperado

## Ejecutar local
```bash
python -m pip install -r requirements.txt
python -m src.download_release --owner Supervision-Inbound --repo IAs
python -m src.main --horizonte 120
