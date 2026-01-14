"""
Purpose-built utility for formatting tables in LaTeX
"""

import re

def no_highlight(name, key):
    return False

class Tabular:
    def __init__(self, columns, highlight_cell=no_highlight):
        self.columns = columns
        self.highlight_cell = highlight_cell
        self.table = []

    def append(self, row):
        self.table.append(row)

    @staticmethod
    def bold(text):
        if res := re.fullmatch(r"\$(.*)\$", text):
            return fr"$\mathbf{{{res[1]}}}$"
        else:
            return fr"\textbf{{{text}}}"

    def fmt_row(self, row, header=False):
        return " & ".join(
            (self.bold(row[col]) if header or self.highlight_cell(row["name"], col) else row[col])
            for col in self.columns.keys()
        ) + " \\\\\n"

    def fmt_table(self):
        colspec = "l" + (len(self.columns)-1) * "c"
        result = fr"\begin{{tabular}}{{{colspec}}}" + "\n\\toprule\n"
        result += self.fmt_row(self.columns, True)
        result += "\\midrule\n"
        for row in self.table:
            result += self.fmt_row(row)
        result += "\\bottomrule\n\\end{tabular}\n"
        return result
