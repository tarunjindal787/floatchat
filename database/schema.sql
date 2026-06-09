-- ============================================================
-- FloatChat PostgreSQL Schema
-- Run: psql -U floatchat -d argo_db -f schema.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── Floats (one row per physical Argo float) ───────────────
CREATE TABLE IF NOT EXISTS floats (
    float_id        VARCHAR(20) PRIMARY KEY,
    platform_number VARCHAR(20) NOT NULL,
    dac             VARCHAR(20),
    ocean_basin     VARCHAR(50),
    wmo_inst_type   VARCHAR(10),
    positioning_sys VARCHAR(20),
    first_seen      DATE,
    last_seen       DATE,
    total_profiles  INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Profiles (one row per profile/cycle) ───────────────────
CREATE TABLE IF NOT EXISTS profiles (
    profile_id      SERIAL PRIMARY KEY,
    float_id        VARCHAR(20) NOT NULL REFERENCES floats(float_id) ON DELETE CASCADE,
    cycle_number    INT NOT NULL,
    juld            TIMESTAMPTZ,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    position_qc     CHAR(1) DEFAULT '0',
    direction       CHAR(1),     -- A=ascending, D=descending
    data_mode       CHAR(1),     -- R=real-time, A=adjusted, D=delayed
    vertical_sampling_scheme VARCHAR(200),
    source_file     VARCHAR(500),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(float_id, cycle_number, direction)
);

CREATE INDEX IF NOT EXISTS idx_profiles_float_id   ON profiles(float_id);
CREATE INDEX IF NOT EXISTS idx_profiles_juld        ON profiles(juld);
CREATE INDEX IF NOT EXISTS idx_profiles_location    ON profiles USING gist(
    ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
);

-- ─── Core Measurements (CTD: T, S, P) ───────────────────────
CREATE TABLE IF NOT EXISTS measurements (
    meas_id         BIGSERIAL PRIMARY KEY,
    profile_id      INT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    pressure        REAL,
    temperature     REAL,
    salinity        REAL,
    temp_adjusted   REAL,
    psal_adjusted   REAL,
    pres_adjusted   REAL,
    temp_qc         CHAR(1) DEFAULT '0',
    psal_qc         CHAR(1) DEFAULT '0',
    pres_qc         CHAR(1) DEFAULT '0'
);

CREATE INDEX IF NOT EXISTS idx_meas_profile_id ON measurements(profile_id);
CREATE INDEX IF NOT EXISTS idx_meas_pressure   ON measurements(pressure);

-- ─── BGC Measurements ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bgc_data (
    bgc_id          BIGSERIAL PRIMARY KEY,
    profile_id      INT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    pressure        REAL,
    doxy            REAL,     -- Dissolved oxygen (µmol/kg)
    doxy_qc         CHAR(1) DEFAULT '0',
    chla            REAL,     -- Chlorophyll-a (mg/m³)
    chla_qc         CHAR(1) DEFAULT '0',
    nitrate         REAL,     -- Nitrate (µmol/kg)
    nitrate_qc      CHAR(1) DEFAULT '0',
    ph_in_situ      REAL,
    ph_qc           CHAR(1) DEFAULT '0',
    bbp700          REAL,     -- Particulate backscattering at 700nm
    bbp700_qc       CHAR(1) DEFAULT '0',
    irradiance_380  REAL,
    irradiance_412  REAL,
    irradiance_490  REAL
);

CREATE INDEX IF NOT EXISTS idx_bgc_profile_id ON bgc_data(profile_id);

-- ─── Ingestion Log ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingestion_log (
    log_id          SERIAL PRIMARY KEY,
    file_path       VARCHAR(500) UNIQUE,
    float_id        VARCHAR(20),
    profiles_count  INT,
    status          VARCHAR(20) DEFAULT 'pending',
    error_msg       TEXT,
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Helper Views ────────────────────────────────────────────
CREATE OR REPLACE VIEW float_summary AS
SELECT
    f.float_id,
    f.dac,
    f.ocean_basin,
    COUNT(DISTINCT p.profile_id)    AS total_profiles,
    MIN(p.juld)                     AS first_obs,
    MAX(p.juld)                     AS last_obs,
    AVG(p.latitude)                 AS mean_lat,
    AVG(p.longitude)                AS mean_lon,
    MIN(p.latitude)                 AS min_lat,
    MAX(p.latitude)                 AS max_lat,
    MIN(p.longitude)                AS min_lon,
    MAX(p.longitude)                AS max_lon
FROM floats f
LEFT JOIN profiles p ON f.float_id = p.float_id
GROUP BY f.float_id, f.dac, f.ocean_basin;

CREATE OR REPLACE VIEW profile_stats AS
SELECT
    p.profile_id,
    p.float_id,
    p.juld,
    p.latitude,
    p.longitude,
    p.cycle_number,
    p.data_mode,
    COUNT(m.meas_id)                AS n_levels,
    MIN(m.pressure)                 AS min_pressure,
    MAX(m.pressure)                 AS max_pressure,
    AVG(m.temperature)              AS mean_temp,
    MIN(m.temperature)              AS min_temp,
    MAX(m.temperature)              AS max_temp,
    AVG(m.salinity)                 AS mean_salinity,
    MIN(m.salinity)                 AS min_salinity,
    MAX(m.salinity)                 AS max_salinity
FROM profiles p
LEFT JOIN measurements m ON p.profile_id = m.profile_id
GROUP BY p.profile_id, p.float_id, p.juld, p.latitude, p.longitude,
         p.cycle_number, p.data_mode;
