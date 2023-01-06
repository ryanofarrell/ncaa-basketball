# %% imports
from itertools import permutations
import sqlite3 as sq

import pandas as pd
import logging
import os
import functools
from datetime import datetime as dt
from typing import Literal


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
    ifExists: Literal["replace", "append", "fail"] = "replace",
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


# %% Logger & Timer
class logger:
    "Logger class"

    def __init__(self, fp, fileLevel, consoleLevel, removeOldFile=False, name="myCustomLogger"):
        format = "%(asctime)s | %(levelname)s | %(message)s"
        if consoleLevel < fileLevel:
            print("Min level set in fileLevel; console will not go lower!")

        if removeOldFile:
            try:
                os.remove(fp)
            except FileNotFoundError:
                # If file exists, do nothing
                pass

        # File logging
        logging.basicConfig(filename=fp, level=fileLevel, format=format)

        # Console Printing
        console = logging.StreamHandler()
        console.setLevel(consoleLevel)
        console.setFormatter(logging.Formatter(format))
        logging.getLogger().addHandler(console)

    def debug(self, message):
        logging.debug(message)

    def info(self, message):
        logging.info(message)

    def warning(self, message):
        logging.warning(message)

    def error(self, message):
        logging.error(message)

    def critical(self, message):
        logging.critical(message)

    def timeFuncInfo(self, fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            try:
                now = dt.now()
                logging.info(f"Fn | {fn.__name__} | Begin")
                result = fn(*args, **kwargs)
                logging.info(f"Fn | {fn.__name__} | Complete | {dt.now() - now}")
                return result
            except Exception as ex:
                logging.critical(f"Exception {ex}")
                raise ex

        return decorated

    def timeFuncDebug(self, fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            try:
                now = dt.now()
                logging.debug(f"Fn | {fn.__name__} | Begin")
                result = fn(*args, **kwargs)
                logging.debug(f"Fn | {fn.__name__} | Complete | {dt.now() - now}")
                return result
            except Exception as ex:
                logging.critical(f"Exception {ex}")
                raise ex

        return decorated


def getRelativeFp(fileDunder, pathToAppend):
    import os
    import pathlib

    fileParentPath = pathlib.Path(fileDunder).parent.absolute()
    newFilePath = os.path.join(fileParentPath, pathToAppend)
    fpParent = pathlib.Path(newFilePath).parent.absolute()
    if not os.path.exists(fpParent):
        os.makedirs(fpParent)
        print(f"Created directory {fpParent}")
    return newFilePath


def get_unique_permutations(cols):
    perms = []
    for i in range(len(cols)):
        perms += ["|".join(sorted(x)) for x in permutations(cols, i)]
    perms = [x.split("|") for x in list(set(perms)) if x != ""]

    return perms


# %% Query in-memory
def qdf(df: pd.DataFrame, q: str):
    with sq.connect(":memory:") as con:
        df.to_sql(
            name="self",
            con=con,
            if_exists="append",
            method="multi",
            chunksize=1000,
        )

        results = pd.read_sql(q, con)

    return results


# %%
