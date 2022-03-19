# %% imports
import sqlite3 as sq

import pandas as pd


def readSql(q: str, db: str = "ncaa.db") -> pd.DataFrame:
    """
    Executes query and returns dataframe of results.

    Args:

        q (str) -- query to execute

        db (str) -- database to run query on; defaults to ncaa.db
    """
    with sq.connect(db) as con:
        df = pd.read_sql(q, con)

    return df


def executeSql(q: str, db: str = "ncaa.db") -> None:
    """
    Runs specified query against database, returning nothing. Use for 'drop table'-like commands.

    Args:

        q (str) -- Query to execute

        db (str) -- Database file. Defaults to ncaa.db

    Returns nothing.
    """
    with sq.connect(db) as con:
        cur = con.cursor()
        cur.execute(q)
        con.commit()


def dfToTable(
    df: pd.DataFrame,
    table: str,
    db: str,
    ifExists: str = "replace",
    indexCols: list[str] | None = None,
) -> None:
    """Saves dataframe as table in sqlite3 database

    Args:

        df (pd.DataFrame) -- data to save

        table (str) -- table name

        db (str) -- database name (ending in .db)

        ifExists (str) -- pass-thru for pandas arg. "replace" (default), "append", "fail"

        indexCols (list of str) -- cols to be used as index. Defaults to None (no index).

    Returns nothing.
    """

    # Handle dtypes
    df = df.convert_dtypes()

    assert ifExists in ["replace", "append", "fail"], f"Invalid ifExists: {ifExists}"

    # Handle index var
    if indexCols is not None:
        index_label = indexCols
        df.set_index(indexCols, drop=True, inplace=True, verify_integrity=True)
        index = True
    else:
        index_label = None
        index = False

    # Load table
    with sq.connect(db) as con:
        df.to_sql(
            name=table,
            con=con,
            if_exists=ifExists,
            method="multi",
            index=index,
            index_label=index_label,
            chunksize=1000,
        )
