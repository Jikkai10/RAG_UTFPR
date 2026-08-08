from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()
import os
URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')   
USER = os.environ.get('NEO4J_USER', 'neo4j')
PASSWORD = os.environ.get('NEO4J_PASSWORD', 'neo4j')


def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Neo4jConnection:
    def __init__(self, uri = URI, user = USER, password = PASSWORD):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        print("Neo4j connection established.")

    def close(self):
        self._driver.close()

    def execute_query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]
    
    def get_driver(self):
        return self._driver


#db = Neo4jConnection(URI, USER, PASSWORD)
