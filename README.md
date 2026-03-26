# Financial Fraud Detection Dashboard

Sistema de análisis y detección de fraude financiero generado con **Claude Code** usando un agente especializado. A partir de un CSV de transacciones, produce un dashboard interactivo en modo oscuro con gráficas de detección de patrones, KPIs y modelo predictivo de Random Forest — todo en un único archivo HTML autocontenido.

---

## Demo

> Carga tu CSV → el dashboard analiza, filtra y visualiza en tiempo real.

| Sección | Descripción |
|---|---|
| KPIs | Tasa de fraude, monto fraudulento, comparativa fraude vs. normal |
| Gráficas | Distribución por tipo, histograma de montos, serie temporal, scatter, heatmap de correlaciones |
| Tabla | Paginada, ordenable, exportable a CSV |
| Modelo | Random Forest con métricas: Precision, Recall, F1, ROC-AUC |

---

## Stack

| Capa | Herramienta |
|---|---|
| Agente IA | Claude Code — agente `financial-fraud-analyzer` |
| Análisis y modelo | Python · pandas · scikit-learn (Random Forest) · matplotlib · seaborn |
| Dashboard | HTML autocontenido · Chart.js 4.x · PapaParse |
| Dataset esperado | CSV con columnas `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud` |

---

## Estructura del repositorio

```
├── fraud_analysis.py              # Script Python: EDA + feature engineering + modelo
├── fraud_dashboard.html           # Dashboard original generado por Claude Code
├── fraud_dashboard_optimized.html # Dashboard optimizado para datasets grandes (6M+ filas)
├── .claude/
│   └── agents/
│       └── financial-fraud-analyzer.md  # Definición del agente Claude Code
└── README.md
```

---

## Cómo usarlo

### 1. Ejecutar el análisis Python

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

# Edita las rutas en fraud_analysis.py:
# FILE_PATH   = 'ruta/a/tu/archivo.csv'
# OUTPUT_HTML = 'ruta/salida/dashboard.html'

python fraud_analysis.py
```

El script recorre 5 fases e imprime un resumen en terminal:

```
[1/5] Cargando datos...
[2/5] Validacion de datos...
[3/5] Realizando Analisis Exploratorio...
[4/5] Detectando patrones de fraude...
[5/5] Entrenando modelo predictivo...
```

### 2. Abrir el dashboard

```bash
open fraud_dashboard_optimized.html   # macOS
start fraud_dashboard_optimized.html  # Windows
```

No requiere servidor. Arrastra tu CSV o haz clic en el área de carga.

### 3. Regenerar con Claude Code

Si tienes Claude Code instalado y el agente configurado:

```
@.claude/agents/financial-fraud-analyzer.md analiza @tu_archivo.csv
```

---

## Optimizaciones de rendimiento

El dashboard original se congelaba con datasets de más de 500K filas. La versión `_optimized` aplica:

| Problema | Solución |
|---|---|
| `Math.max(...array)` con millones de filas — explota el call stack | Un único `for` que calcula los tres máximos en un pase |
| 10+ `filteredData.filter()` por cada actualización de gráfica | Índice pre-calculado `buildIndex()` — los charts leen en O(1) |
| `calculateCorrelation()` hacía 2 `reduce()` por par (72 pases para el heatmap) | Columnas extraídas en `Float64Array` una vez; un pase por par |
| `chunk` de PapaParse vacío — cargaba todo de golpe | Streaming real en bloques de 512 KB con barra de progreso |
| Animaciones de Chart.js a 500–800ms bloqueando el hilo | `animation: { duration: 0 }` en todos los charts |
| `updateSuspicionList` hacía `.filter().sort()` en millones de filas | Mini-heap top-K en un único pase |
| Sliders disparando recomputes en cada tick | Debounce de 120 ms en `applyFilters` |

Resultado: carga fluida en datasets de 6M+ filas.

---

## Patrones de fraude que detecta

El análisis identifica automáticamente:

- **Discrepancias de balance** — transacciones donde el cambio de saldo no coincide con el monto
- **Vaciado de cuenta** — `newbalanceOrig == 0` tras la transacción
- **Transacciones de alto valor** — por encima del percentil 95 del dataset
- **Frecuencia por origen** — cuentas con velocidad de transacción anómala
- **Tipo de destino** — merchants vs. clientes como receptores de fondos
- **Distribución temporal** — steps con concentración inusual de fraude

---

## Modelo predictivo

Random Forest con `class_weight='balanced'` para manejar el desbalanceo de clases (fraude < 0.2% en el dataset de referencia).

**Features usadas:**

```python
['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
 'dest_is_merchant', 'dest_is_customer', 'high_value', 'empties_account',
 'balance_error_orig', 'balance_error_dest', 'type_encoded']
```

**Métricas típicas sobre el dataset de referencia (6.3M transacciones):**

| Métrica | Valor |
|---|---|
| ROC-AUC | ~0.98 |
| Precision | ~0.94 |
| Recall | ~0.76 |
| F1-Score | ~0.84 |

---

## Dataset de referencia

El proyecto fue desarrollado y validado sobre el dataset público **PaySim** (`PS_20174392719_1491204439457_log.csv`), una simulación de transacciones móviles de dinero basada en datos reales del operador financiero africano M-Pesa.

- 6.3M transacciones · 11 columnas · tasa de fraude: 0.13%
- Solo los tipos `TRANSFER` y `CASH_OUT` contienen fraudes reales
- Disponible en [Kaggle — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
