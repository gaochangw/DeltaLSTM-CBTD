import warnings
import pandas as pd


class PandasLogger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.list_header = []
        self.loglist = []

    def add_row(self, list_header, list_value):
        self.list_header = list_header
        row = {}
        for header, value in zip(list_header, list_value):
            row[header] = value
        self.loglist.append(row)

    def write_csv(self, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            df = pd.DataFrame(self.loglist, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)

    def write_log(self, log_stat):
        # Evaluate Weight Range
        # for name, param in net.named_parameters():
        #     param_data = param.data
        #     print("Name: %30s | Min: %f | Max: %f" % (name, torch.min(param_data), torch.max(param_data)))

        # Create Log List
        list_log_headers = []
        list_log_values = []
        for k, v in log_stat.items():
            list_log_headers.append(k)
            list_log_values.append(v)

        # Write Log
        self.add_row(list_log_headers, list_log_values)
        self.write_csv()

    def write_log_idx(self, idx, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            loglist_best = [self.loglist[idx]]
            df = pd.DataFrame(loglist_best, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)
