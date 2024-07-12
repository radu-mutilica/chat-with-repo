import os

from pymongo import MongoClient

from libs.models import RepoMetadata

mongo_connection_string = f'mongodb://{os.environ["MONGO_HOST"]}:{os.environ["MONGO_PORT"]}/stats'


class CrawlStats:
    """
    A class for ORM bindings for crawl statistics and operations.
    """

    def __init__(self, connection_string=mongo_connection_string):
        self.client = MongoClient(connection_string)
        self.db = self.client.get_default_database()
        self.collection = self.db['stats']

    def update_finished_crawl(self, stats, collection_name):
        """Save the finished crawl stats to the database.

        Args:
            stats (dict): A dictionary containing the statistics of the finished crawl.
            collection_name (str): The name of the collection in the database where the statistics
            should be logged.

        Returns:
            ObjectId: The generated unique identifier of the inserted document.
        """
        collection = self.db[collection_name]
        result = collection.insert_one(stats)
        return result.inserted_id

    def get_last_commit(self, repo):
        """Get the timestamp of the last commit to a repository.

        Args:
            repo (str): The name of the repository.

        Returns:
            int: The timestamp of the last commit made to the repository.
            If no commit exists, it returns 0.
        """
        repo_crawl_stats = self.collection.find_one(
            {'_id': repo},
        )

        if repo_crawl_stats:
            return RepoMetadata.model_validate(repo_crawl_stats)
        else:
            return None

    def update_crawl_stats(self, repo, metadata):
        """Update the stats for this repository.

        Repo name.
        Repo picture.
        The repo's description (if it has one).
        Last time it was updated by the crawler.
        Time the repo was added to our list of repos.
        A unique id for that repo.

        Args:
            repo (str): The name or identifier of the repository.
            metadata (RepoMetadata): the metadata info for this repo/branch combo.

        """
        self.collection.update_one(
            {
                '_id': repo
            },
            {
                '$set': metadata.model_dump()
            },
            upsert=True
        )

    def get_repos(self):
        """Get the list of repositories, together will their crawl stats"""
        return self.collection.find()
