# FloatChat 

> **AI-Powered Conversational Interface for ARGO Ocean Data Discovery and Visualization**

FloatChat is an end-to-end system that ingests ARGO oceanographic float data (NetCDF), stores it in PostgreSQL + a vector database (ChromaDB/FAISS), and exposes a conversational Streamlit dashboard powered by RAG + LLM-based Text-to-SQL via the **Model Context Protocol (MCP)**.

---

##  Features

| Feature | Description |
|---------|-------------|
|  **Natural Language Queries** | Ask questions like *"Show salinity profiles in the Arabian Sea in March 2023"* |
| **Geospatial Visualization** | Float trajectories, profile heatmaps, multi-float overlays (Plotly Mapbox) |
|  **Ocean Profile Plots** | CTD vertical profiles, T-S diagrams, Hovmöller depth-time sections |
|  **BGC Dashboard** | Dissolved oxygen, chlorophyll-a, nitrate, pH, backscatter panels |
|  **RAG + Text-to-SQL** | LLM translates natural language → SQL; vector search over profile summaries |
|  **MCP Server** | 7 typed tools exposed via Model Context Protocol for LLM agents |
|  **Data Export** | CSV, CF-compliant NetCDF, ASCII table downloads |
|  **SQL Safety** | `sqlglot`-based parsing — only `SELECT` statements allowed |
|  **Docker-ready** | One-command startup with Docker Compose |

---

##  Architecture

```
ARGO NetCDF Files
       │
       ▼
┌─────────────────┐     ┌─────────────────────┐
│  Ingestion      │────▶│  PostgreSQL (pgvector)│
│  Pipeline       │     │  floats / profiles   │
│  (xarray/pandas)│     │  measurements / bgc  │
└─────────────────┘     └─────────────────────┘
       │                          │
       ▼                          ▼
┌─────────────────┐     ┌─────────────────────┐
│  ChromaDB       │     │  SQL Executor        │
│  Vector Store   │     │  (safe, validated)   │
│  (embeddings)   │     └─────────────────────┘
└─────────────────┘              │
       │                         ▼
       └──────────────┐  ┌───────────────────┐
                      │  │  LLM Backend       │
                      └─▶│  (GPT/Mistral/    │
                         │   QWEN/LLaMA)      │
                         └───────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Streamlit Dashboard     │
                    │  Chat │ Map │ Profiles   │
                    │  BGC  │ Export          │
                    └─────────────────────────┘
```

---

##  Quick Start

### Option 1 — Docker Compose (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/your-org/floatchat.git
cd floatchat

# 2. Copy and configure environment
cp .env.example .env
# Edit .env — set LLM_PROVIDER, model, and API keys

# 3. Start all services
docker-compose up -d

# 4. Download sample Indian Ocean data
python scripts/download_sample_data.py --n-floats 20

# 5. Run ingestion
docker exec floatchat_app python -m ingestion.pipeline \
    --input data/raw/ --region indian_ocean --init-schema

# 6. Open the app
# → http://localhost:8501
```

### Option 2 — Local Development

```bash
# 1. Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL + ChromaDB
docker-compose up -d postgres chromadb

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Initialize DB schema
python -m ingestion.pipeline --init-schema

# 6. Download + ingest data
python scripts/download_sample_data.py --n-floats 20
python -m ingestion.pipeline --input data/raw/ --region indian_ocean

# 7. (Optional) Start MCP server
python -m backend.mcp_server &

# 8. Launch Streamlit
streamlit run frontend/app.py
```

---

##  LLM Configuration

Set `LLM_PROVIDER` in `.env` to switch between models:

| Provider | `.env` setting | Notes |
|----------|---------------|-------|
| **Ollama** (default) | `LLM_PROVIDER=ollama` | Free, local, no API key. Install [Ollama](https://ollama.ai) + `ollama pull mistral` |
| **OpenAI** | `LLM_PROVIDER=openai` | Best quality. Needs `OPENAI_API_KEY` |
| **Together.ai** | `LLM_PROVIDER=together` | Cloud hosted open models |
| **Groq** | `LLM_PROVIDER=groq` | Ultra-fast inference, LLaMA-3 |

### Local (Ollama) Setup — No API Key Needed

```bash
# Install Ollama: https://ollama.ai
ollama pull mistral          # 4GB, good quality
# or
ollama pull llama3.2         # 2GB, faster
```

Then set in `.env`:
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
```

---

##  Project Structure

```
floatchat/
├── ingestion/
│   ├── netcdf_parser.py      # ARGO NetCDF → DataFrames
│   ├── db_writer.py          # PostgreSQL batch upsert
│   ├── vector_indexer.py     # ChromaDB embedding + search
│   ├── parquet_exporter.py   # Partitioned Parquet cache
│   └── pipeline.py           # CLI orchestrator
│
├── backend/
│   ├── mcp_server.py         # MCP server (7 tools)
│   ├── query_router.py       # Intent classification
│   ├── text_to_sql.py        # NL → SQL with safety layer
│   ├── rag_retriever.py      # Vector semantic search
│   ├── sql_executor.py       # Safe SQL execution
│   ├── response_generator.py # Final LLM response builder
│   └── models/
│       └── llm_client.py     # Unified LLM API client
│
├── frontend/
│   ├── app.py                # Streamlit main app
│   ├── components/
│   │   ├── map_view.py       # Plotly Mapbox charts
│   │   ├── profile_plot.py   # CTD / T-S / Hovmöller
│   │   ├── bgc_comparison.py # BGC panels
│   │   └── data_export.py    # CSV / NetCDF / ASCII
│   └── assets/style.css
│
├── database/schema.sql       # PostgreSQL schema
├── scripts/
│   └── download_sample_data.py  # GDAC data downloader
├── tests/                    # pytest test suite
├── config/settings.py        # Pydantic settings
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

##  Example Queries

| Query | Intent |
|-------|--------|
| *"Show temperature profiles in the Arabian Sea in March 2023"* | SQL → CTD profile plot |
| *"Compare salinity in Bay of Bengal vs Arabian Sea"* | SQL → comparison chart |
| *"What are the nearest Argo floats to 12°N 72°E?"* | Nearest float search |
| *"Show trajectory of float 6902742"* | Trajectory map |
| *"What is the Argo BGC program?"* | RAG explanation |
| *"Export float 2902733 data as NetCDF"* | Data export |
| *"Show chlorophyll-a at the surface for the last 6 months"* | SQL → BGC time series |

---

## 🔌 MCP Tools

The MCP server exposes these tools for LLM agents:

| Tool | Description |
|------|-------------|
| `search_profiles` | Semantic search over profile summaries |
| `execute_sql_query` | Run validated SELECT SQL |
| `natural_language_to_sql` | NL → SQL → results |
| `get_float_trajectory` | Lat/lon time series for a float |
| `get_nearest_floats` | N nearest floats to coordinates |
| `get_ctd_profile` | Full CTD data for a profile |
| `get_bgc_profile` | BGC data for a profile |

---

##  Running Tests

```bash
# Run all tests
pytest

# Run without DB/LLM (unit tests only)
pytest -m "not integration"

# Run with coverage report
pytest --cov=ingestion --cov=backend --cov-report=html
```

---

##  Database Schema

```sql
floats       → one row per physical Argo float
profiles     → one row per profile/cycle (lat, lon, date)
measurements → CTD levels (pressure, temperature, salinity)
bgc_data     → BGC levels (O2, Chl-a, NO3, pH, bbp)
```

PostGIS spatial index on `profiles(latitude, longitude)` enables efficient nearest-float queries.

---

## 🗺 Supported Visualizations

| Chart | Use Case |
|-------|---------|
| Float trajectory map | Drift paths over time |
| Profile heatmap | Spatial density, coloured by T/S/depth |
| CTD vertical profile | Water column temperature + salinity |
| T-S diagram | Water mass identification |
| Hovmöller section | Depth-time evolution of T/S |
| BGC multi-panel | O₂, Chl-a, NO₃ vertical profiles |
| BGC time series | Surface/subsurface/deep layer trends |
| Regional box plot | Variable distribution comparison |

---

##  Data Sources

- **Primary**: [Argo Global Data Assembly Centre (GDAC)](https://argo.ucsd.edu/data/)
- **Mirror 1**: [Ifremer GDAC](https://data-argo.ifremer.fr/)
- **Mirror 2**: [US GODAE](https://usgodae.org/ftp/outgoing/argo/)
- **PoC Data**: Indian Ocean floats (DACs: INCOIS, CSIO, Coriolis)

---

##  Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend: `openai`, `ollama`, `together`, `groq` |
| `LLM_MODEL` | `mistral` | Model name |
| `OPENAI_API_KEY` | — | Required if `LLM_PROVIDER=openai` |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai` |
| `MAX_SQL_ROWS` | `5000` | Maximum rows returned per SQL query |
| `RAG_TOP_K` | `8` | Number of vector search results |

---

##  License

MIT License — see [LICENSE](LICENSE) for details.

---

##  Acknowledgements

- [Argo Program](https://argo.ucsd.edu/) — global ocean observing system
- [Ifremer](https://www.ifremer.fr/) — Argo GDAC host
- [INCOIS](https://www.incois.gov.in/) — Indian Ocean Argo data
- [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), [LangChain](https://langchain.com/)
- [ChromaDB](https://www.trychroma.com/), [pgvector](https://github.com/pgvector/pgvector)
