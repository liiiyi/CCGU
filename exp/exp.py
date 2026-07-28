import json
import logging
import os

from lib_dataset.data_store import DataStore


class Exp:
    def __init__(self, args):
        self.logger = logging.getLogger('exp')

        self.args = args
        self.data_store = DataStore(args)

    def load_data(self):
        pass

    def write_metrics(self, metrics):
        """Persist this stage's metrics as JSON when ``--metrics_file`` is given.

        Keeps the run's numbers machine-readable so the reproduction tables are
        built from recorded values rather than re-read from logs.
        """
        path = self.args.get('metrics_file')
        if not path:
            return None
        payload = dict(metrics)
        payload['config'] = {
            key: self.args[key]
            for key in sorted(self.args)
            if key not in ('metrics_file', 'log_file')
        }
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        self.logger.info('metrics written to %s', path)
        return path
