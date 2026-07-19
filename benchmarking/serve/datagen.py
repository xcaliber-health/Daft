"""Generate TPC-H data and load it into an Iceberg warehouse.

Data is produced by an engine-neutral generator and written through the
canonical table-format writer, so neither benchmarked engine reads a layout
produced by the other.
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import pyarrow as pa

logger = logging.getLogger(__name__)

TPCH_TABLES = (
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
)


def generate_warehouse(warehouse: pathlib.Path, scale_factor: float) -> dict[str, str]:
    """Generate TPC-H tables at the given scale into an Iceberg warehouse.

    Parameters
    ----------
    warehouse:
        Directory for the catalog database and table data.
    scale_factor:
        TPC-H scale factor; ``0.01`` is suitable for smoke runs.

    Returns:
    -------
    dict[str, str]
        Mapping of table name to its current metadata file location.
    """
    from pyiceberg.catalog.sql import SqlCatalog

    import duckdb

    warehouse.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "bench",
        uri=f"sqlite:///{warehouse}/catalog.db",
        warehouse=f"file://{warehouse}",
    )
    if ("tpch",) not in catalog.list_namespaces():
        catalog.create_namespace("tpch")

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={scale_factor})")

    locations: dict[str, str] = {}
    for name in TPCH_TABLES:
        arrow_table = con.execute(f"SELECT * FROM {name}").fetch_arrow_table()
        # Monetary columns are generated as decimals; store them as doubles so
        # both engines evaluate the same arithmetic (decimal multiplication in
        # the query set overflows fixed-precision bounds).
        schema = arrow_table.schema
        for index, field in enumerate(schema):
            if pa.types.is_decimal(field.type):
                arrow_table = arrow_table.set_column(index, field.name, arrow_table.column(index).cast(pa.float64()))
        # Uppercase column names match the DataFrame query set; SQL engines
        # resolve identifiers case-insensitively.
        arrow_table = arrow_table.rename_columns([c.upper() for c in arrow_table.column_names])
        identifier = f"tpch.{name}"
        if catalog.table_exists(identifier):
            catalog.drop_table(identifier)
        table = catalog.create_table(identifier, schema=arrow_table.schema)
        table.append(arrow_table)
        locations[name] = table.metadata_location
        logger.info("loaded %s: %d rows", identifier, arrow_table.num_rows)

    return locations


def existing_locations(warehouse: pathlib.Path) -> dict[str, str]:
    """Read table metadata locations from an already-generated warehouse.

    Parameters
    ----------
    warehouse:
        Directory previously populated by :func:`generate_warehouse`.

    Returns:
    -------
    dict[str, str]
        Mapping of table name to its current metadata file location.
    """
    from pyiceberg.catalog.sql import SqlCatalog

    catalog = SqlCatalog(
        "bench",
        uri=f"sqlite:///{warehouse}/catalog.db",
        warehouse=f"file://{warehouse}",
    )
    return {name: catalog.load_table(f"tpch.{name}").metadata_location for name in TPCH_TABLES}


def main() -> None:
    """Command-line entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Generate a TPC-H Iceberg warehouse")
    parser.add_argument("--warehouse", required=True, help="output directory")
    parser.add_argument("--scale-factor", type=float, default=0.01)
    args = parser.parse_args()
    generate_warehouse(pathlib.Path(args.warehouse), args.scale_factor)


if __name__ == "__main__":
    main()
