CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
	email TEXT NOT NULL,
    password TEXT NOT NULL
, company TEXT);
CREATE TABLE IF NOT EXISTS "sim" (
	"id"	INTEGER,
	"id_user"	INTEGER,
	"created_at"	TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	"company"	TEXT NOT NULL DEFAULT NULL,
	"line"	TEXT DEFAULT NULL,
	"day_type"	TEXT DEFAULT NULL,
	"battery_size"	FLOAT,
	"max_charging_power"	INTEGER, elevation_deposit INTEGER, elevation_starting_station INTEGER, elevation_arrival_station INTEGER, capex_pars TEXT, opex_pars TEXT, max_charging_powers TEXT,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("id_user") REFERENCES "user"("id")
);
CREATE TABLE IF NOT EXISTS "bus_model" (
	"id"	INTEGER,
	"code"	TEXT DEFAULT NULL,
	"name"	TEXT DEFAULT NULL,
	"features"	TEXT DEFAULT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "terminal" (
"id" INTEGER,
  "name" TEXT,
  "company" TEXT,
  "elevation_m" INTEGER,
  "is_charging_station" INTEGER
);
CREATE TABLE IF NOT EXISTS "distance" (
"id_starting_station" INTEGER,
  "id_arrival_station" INTEGER,
  "distance_km" REAL,
  "avg_travel_time_min" INTEGER,
  "company" TEXT
);
