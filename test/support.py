from collections import defaultdict, OrderedDict

import pandas as pd


class ResultCollector(object):

    def __init__(self):
        self.results = defaultdict(OrderedDict)

    def collect(self, scenario, env, result):
        self.results[scenario.name][env.name] = result

    def finalize(self):
        scenarios = list(self.results.keys())
        envs = list(self.results[scenarios[0]].keys())


        with open("report.html", "w") as fp:
            fp.write("<html><body>")
            for scenario in scenarios:
                fp.write("<h1>%s</h1>"%scenario)

                results = [self.results[scenario][env] for env in envs]
                df = pd.DataFrame(results, columns=results[0]._fields)
                df["env"] = envs
                df.set_index("env", inplace=True)

                fp.write(df.to_html())

            fp.write("</body></html>")