"""
Diese Modul stellt Klassen zur Verfügung, um auf Datenbanken, speziell auf Oracle-Datenbanken, zuzugreifen.
"""

import os

from attribmod.utils import configuration, my_logging
from sqlalchemy import create_engine, engine, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.pool import NullPool

LOGGER = my_logging.logger(os.path.basename(__file__))


class DBConnection:
    """
    Diese Klasse stellt eine Verbindung zu einer Datenbank her und ermöglicht das Ausführen von Statements.
    """

    def __init__(self, connection_url, transaction=False):
        if isinstance(connection_url, str):
            self.connection_url = [connection_url]
        else:
            self.connection_url = connection_url
        self._transaction = transaction
        self._connection = None

    def _connect(self):
        dberror = None
        i = 0
        for url in self.connection_url:
            i += 1
            try:
                dbengine = create_engine(url, poolclass=NullPool, max_identifier_length=128)
                dbengine.connect()
            except DatabaseError as err:
                dberror = err
                LOGGER.warning(
                    'Verbindungsversuch zur Datenbank "%s" ist fehlgeschlagen (%i/%i)',
                    url,
                    i,
                    len(self.connection_url),
                )
            else:
                self._connection = dbengine.connect()
                if self._transaction:
                    self._connection.begin()
                LOGGER.debug(
                    'Verbindung zur Datenbank wurde hergestellt (%i/%i)',
                    i,
                    len(self.connection_url),
                )
                return
        if dberror is not None:
            raise dberror
        raise RuntimeError('Could not establish database connection')

    def execute(self, sql: str):
        """
        Führt ein SQL-Statement aus

        Args:
            sql: Das SQL-Statement

        Returns:
            Gibt das Ergebnis zurück
        """

        connection = self.get_connection()
        sql = text(sql)
        try:
            return connection.execute(sql)
        except DatabaseError as err:
            if err.code == '4xp6' and (err.connection_invalidated or 'ORA-25408' in err.args[0]):
                LOGGER.warning(
                    'Verbindung ist invalidiert oder ORA-25408 ist aufgetreten. Versuche es erneut.'
                )
                connection.close()
                connection = self.get_connection()
                return connection.execute(sql)
            raise err

    def get_connection(self) -> engine:
        """
        Gibt die SQLAlchemy-Engine zurück.

        Returns:
            SQLAlchemy-Enginge
        """
        if self._connection is None:
            self._connect()
        return self._connection

    def drop_table(self, table: str):
        """
        Löscht eine Tabelle

        Args:
            table: Tabellenname inkl. Schema

        Returns:

        """
        if 'oracle' in self.get_connection().driver:
            sql = (
                '''
                BEGIN
                    EXECUTE IMMEDIATE 'DROP TABLE %s';
                EXCEPTION
                    WHEN OTHERS THEN
                        IF SQLCODE != -942 THEN
                            RAISE;
                        END IF;
                END;'''
                % table
            )
            self.get_connection().execute(sql)
        else:
            sql = 'drop table if exists %s' % table
            self.get_connection().execute(sql)

    def table_exists(self, tablename: str) -> bool:
        """
        Gibt True zurück, falls die Tabelle `tablename` existiert, ansonsten False
        Der Tabellenname kann eine Schemabezeichnung enthalten.

        Args:
            tablename: Name der Tabelle mit optionaler Schemaangabe

        Returns:
            Ob die Tabelle existiert oder nicht
        """
        if '.' in tablename:
            [owner, tablename] = tablename.split('.')
        else:
            raise ValueError('tablename needs tablespace: ', tablename)
        sql = (
            "select count(*) from all_objects where object_type in ('TABLE','VIEW') "
            "and object_name = '%s' and owner = '%s'" % (tablename, owner)
        )
        result = self.get_connection().execute(sql).fetchone()[0]
        return bool(result)


class DWH(DBConnection):
    """
    Liest aus der Konfigurationsdatei die Zugangsdaten zu einer Oracle-Datenbank und verbindet sich mit dieser.
    """

    def __init__(self, user: str, transaction=False):
        super().__init__(configuration.get_value('database', user), transaction=transaction)
        if (
            configuration.get_value('database', 'oracle_instantclient') is not None
            and 'instantclient' not in os.environ['PATH']
        ):
            os.environ['PATH'] = os.path.join(
                os.environ['PATH'], configuration.get_value('database', 'oracle_instantclient')
            )
            LOGGER.debug(
                'Pfad wurde um Oracle Instantclient ergänzt (%s)',
                configuration.get_value('database', 'oracle_instantclient'),
            )


class Postgres(DBConnection):
    def __init__(self, user: str):
        super().__init__(configuration.get_value('database', user))
