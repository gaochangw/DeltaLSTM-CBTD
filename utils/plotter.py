import csv
import os
import warnings
from glob import glob
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    def __init__(self, plot_dir, data_dir, name):
        self.plot_dir = plot_dir
        self.data_dir = data_dir
        self.name = name
        self.df = None
        self.file_list = []  # List of CSV file paths for plotting
        self.legend_list = []  # List of CSV file paths for plotting
        # self.dict_abbr2name = {'SD': 'Seed',
        #                        'FT': 'Feature_Type',
        #                        'NF': 'Num_Channels',
        #                        'FL': 'Low_Frequency',
        #                        'FH': 'High_Frequency',
        #                        'Q': 'Quality_Factor',
        #                        'SP': 'Channel_Spacing',
        #                        'FBW': 'Feature_BitWidth',
        #                        'FG': 'Feature_Gradient',
        #                        'CLA': 'Classifier',
        #                        'FC': 'Add_FC',
        #                        'L-Val': 'Validation_Loss',
        #                        'L-Test': 'Test_Loss',
        #                        'ACC-VAL': 'Validation_Accuracy',
        #                        'ACC-TEST': 'Test_Accuracy',
        #                        'NUM_ARRAY_PE': 'M',
        #                        }

    def add_file(self, file_list, legend_list=None):
        self.file_list.append(file_list)
        if legend_list is not None:
            self.legend_list.extend(legend_list)
            if len(self.file_list) != len(self.legend_list):
                warnings.warn("Length of file list & legend list must be the same...", RuntimeWarning)
        else:
            self.legend_list = None

    def filter_data(self, dict_cond_cols):
        # Select Data
        if dict_cond_cols is not None:
            condition = True
            for k, v in dict_cond_cols.items():
                if type(k) == str:
                    if type(v[1]) == str:
                        str_eval = '(self.df.' + k + v[0] + '\'' + str(v[1]) + '\'' + ')'
                    else:
                        str_eval = '(self.df.' + k + v[0] + str(v[1]) + ')'
                else:
                    n = 0
                    str_eval = ''
                    for col_name, cond in zip(k, v):
                        if n > 0:
                            str_eval += ' | '
                            # print(str_eval)
                        if type(cond[1]) == str:
                            str_eval += '(self.df.' + col_name + cond[0] + '\'' + str(cond[1]) + '\'' + ')'
                        else:
                            str_eval += '(self.df.' + col_name + cond[0] + str(cond[1]) + ')'
                        # print(str_eval)
                        n += 1

                condition &= eval(str_eval)
            df = self.df[condition]
            return df

    def bar_plot(self, x, y, hue=None, col=None, order=None, dict_cond_cols=None, ylim=None, ystep=None, figsize=None,
                 figname=None):
        from matplotlib import pyplot
        sns.set_context('paper')
        sns.set(style="darkgrid")
        sns.set(font_scale=1)

        # Set Plot Size
        fig, ax = pyplot.subplots(figsize=figsize)

        # # Translate Inputs
        # try:
        #     x = self.dict_abbr2name[x]
        # except:
        #     pass
        #
        # try:
        #     y = self.dict_abbr2name[y]
        # except:
        #     pass
        #
        # try:
        #     hue = self.dict_abbr2name[hue]
        # except:
        #     pass
        #
        # try:
        #     col = self.dict_abbr2name[col]
        # except:
        #     pass

        # Select Data
        if dict_cond_cols is not None:
            condition = True
            for k, v in dict_cond_cols.items():
                # try:
                #     k = self.dict_abbr2name[k]
                # except:
                #     pass

                if type(k) == str:
                    if type(v[1]) == str:
                        str_eval = '(self.df.' + k + v[0] + '\'' + str(v[1]) + '\'' + ')'
                    else:
                        str_eval = '(self.df.' + k + v[0] + str(v[1]) + ')'
                else:
                    n = 0
                    str_eval = ''
                    for col_name, cond in zip(k, v):
                        if n > 0:
                            str_eval += ' | '
                            # print(str_eval)
                        if type(cond[1]) == str:
                            str_eval += '(self.df.' + col_name + cond[0] + '\'' + str(cond[1]) + '\'' + ')'
                        else:
                            str_eval += '(self.df.' + col_name + cond[0] + str(cond[1]) + ')'
                        # print(str_eval)
                        n += 1

                print(str_eval)
                condition &= eval(str_eval)
            df = self.df[condition]
        print(df)

        # Save Name
        def gen_fig_name(dict_cond_cols):
            figname = 'X=' + x + '#Y=' + y
            if col is not None:
                figname += '#Col=' + col
            elif hue is not None:
                figname += '#Hue=' + hue
            else:
                figname = 'X=' + x + '#Y=' + y
            for k, v in dict_cond_cols.items():
                if type(k) == str:
                    figname += '#' + k + v[0] + str(v[1])
                else:
                    n = 0
                    figname += '#'
                    for col_name, cond in zip(k, v):
                        if n > 0:
                            figname += '_or_'
                        if type(cond[1]) == str:
                            figname += '(self.df.' + col_name + cond[0] + '\'' + str(cond[1]) + '\'' + ')'
                        else:
                            figname += '(self.df.' + col_name + cond[0] + str(cond[1]) + ')'
                        n += 1
            return figname

        savename = gen_fig_name(dict_cond_cols)

        # create plot
        if col is not None:
            barplot = sns.catplot(x=x, y=y, hue=hue, col=col, data=df,
                                  # palette='hls',
                                  palette='rocket',
                                  kind='bar'
                                  # palette=cm.Blues(df['b'] * 10)
                                  # order=['male', 'female'],
                                  # capsize=0.05,
                                  # saturation=8,
                                  # errcolor='gray',
                                  # errwidth=2,
                                  # ci='sd'
                                  )
            # barplot.set(xlim=(0, 60), ylim=(0, 12), xticks=[10, 30, 50], yticks=[2, 6, 10]).fig.subplots_adjust(wspace=.05, hspace=.05)
            barplot.set(ylim=ylim)
            if figname is not None:
                savename = figname

            for axlist in barplot.axes:
                for axes in axlist:
                    for p in axes.patches:
                        axes.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                                      ha='center', va='center', fontsize=13, color='red', xytext=(0, 20),
                                      textcoords='offset points')

            # barplot.savefig(self.plot_dir + savename + '.png', format='png', bbox_inches='tight')

        else:
            barplot = sns.barplot(x=x,
                                  y=y,
                                  hue=hue,
                                  data=df,
                                  # order=order,
                                  ax=ax,
                                  # palette='hls',
                                  # palette='rocket',
                                  # palette=cm.Blues(df['b'] * 10)
                                  # order=['male', 'female'],
                                  capsize=0.05,
                                  saturation=8,
                                  errcolor='gray',
                                  errwidth=2,
                                  ci='sd'
                                  )
            axes = barplot.axes
            if ylim is not None and ystep is not None:
                axes.set_ylim(ylim[0], ylim[1])
                axes.set_yticks(np.arange(ylim[0], ylim[1], (ylim[1] - ylim[0]) / ystep))
            axes.set_title(savename)

            for p in axes.patches:
                axes.annotate("%.4f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='center', fontsize=13, color='red', xytext=(0, 20),
                              textcoords='offset points')

            if figname is not None:
                savename = figname
            # fig.savefig(self.plot_dir + savename + '.png', format='png', bbox_inches='tight')
        # plt.show()

    def line_plot(self, x, y, hue=None, col=None, style=None, dict_cond_cols=None, ylim=None, ystep=None, figsize=None,
                  figname=None, err_style='bars', label=None, palette=None, fig=None, ax=None, marker=None, color=None):
        from matplotlib import pyplot
        # sns.set_context('paper')
        # sns.set(style="darkgrid")
        # sns.set(font_scale=1)

        # Set Plot Size
        if fig and ax is None:
            fig, ax = pyplot.subplots(figsize=figsize)

        # Select Data
        df = self.df
        if dict_cond_cols is not None:
            condition = True
            for k, v in dict_cond_cols.items():
                if type(k) == str:
                    if type(v[1]) == str:
                        str_eval = '(self.df.' + k + v[0] + '\'' + str(v[1]) + '\'' + ')'
                    else:
                        str_eval = '(self.df.' + k + v[0] + str(v[1]) + ')'
                else:
                    n = 0
                    str_eval = ''
                    for col_name, cond in zip(k, v):
                        if n > 0:
                            str_eval += ' | '
                            # print(str_eval)
                        if type(cond[1]) == str:
                            str_eval += '(self.df.' + col_name + cond[0] + '\'' + str(cond[1]) + '\'' + ')'
                        else:
                            str_eval += '(self.df.' + col_name + cond[0] + str(cond[1]) + ')'
                        # print(str_eval)
                        n += 1

                condition &= eval(str_eval)
            df = self.df[condition]

        # Save Name
        def gen_fig_name(dict_cond_cols):
            figname = 'X=' + x + '#Y=' + y
            if col is not None:
                figname += '#Col=' + col
            elif hue is not None:
                figname += '#Hue=' + hue
            elif style is not None:
                figname += '#Style=' + str(style)

            for k, v in dict_cond_cols.items():
                if type(k) == str:
                    figname += '#' + k + v[0] + str(v[1])
                else:
                    n = 0
                    figname += '#'
                    for col_name, cond in zip(k, v):
                        if n > 0:
                            figname += '_or_'
                        if type(cond[1]) == str:
                            figname += '(self.df.' + col_name + cond[0] + '\'' + str(cond[1]) + '\'' + ')'
                        else:
                            figname += '(self.df.' + col_name + cond[0] + str(cond[1]) + ')'
                        n += 1
            return figname

        savename = gen_fig_name(dict_cond_cols)

        # create plot
        g = sns.lineplot(x=x,
                         y=y,
                         hue=hue,
                         data=df,
                         ax=ax,
                         label=label,
                         # palette='hls',
                         style=style,
                         err_style=err_style,
                         # palette='rocket',
                         # palette=cm.Blues(df['b'] * 10)
                         ci='sd',
                         # style="event",
                         palette=palette,
                         dashes=False,
                         markers=True,
                         markersize=10,
                         marker=marker,
                         linewidth=3,
                         color=color
                         )

    def heatmap(self, x, y, value, dict_cond_cols=None, ylim=None, ystep=None, figsize=None,
                figname=None, fig=None, ax=None):
        from matplotlib import pyplot
        # sns.set_context('paper')
        # sns.set(style="darkgrid")
        # sns.set(font_scale=1)

        # Set Plot Size
        if fig and ax is None:
            fig, ax = pyplot.subplots(figsize=figsize)

        # Translate Inputs
        try:
            x = self.dict_abbr2name[x]
        except:
            pass

        try:
            y = self.dict_abbr2name[y]
        except:
            pass

        # Select Data
        df = self.df
        if dict_cond_cols is not None:
            condition = True
            for k, v in dict_cond_cols.items():
                try:
                    k = self.dict_abbr2name[k]
                except:
                    pass

                if type(k) == str:
                    if type(v[1]) == str:
                        str_eval = '(self.df.' + k + v[0] + '\'' + str(v[1]) + '\'' + ')'
                    else:
                        str_eval = '(self.df.' + k + v[0] + str(v[1]) + ')'
                else:
                    n = 0
                    str_eval = ''
                    for col_name, cond in zip(k, v):
                        if n > 0:
                            str_eval += ' | '
                            # print(str_eval)
                        if type(cond[1]) == str:
                            str_eval += '(self.df.' + col_name + cond[0] + '\'' + str(cond[1]) + '\'' + ')'
                        else:
                            str_eval += '(self.df.' + col_name + cond[0] + str(cond[1]) + ')'
                        # print(str_eval)
                        n += 1

                condition &= eval(str_eval)
            df = self.df[condition]

        # Save Name
        def gen_fig_name(dict_cond_cols):
            figname = 'X=' + x + '#Y=' + y + '#Value=' + value

            for k, v in dict_cond_cols.items():
                if type(k) == str:
                    figname += '#' + k + v[0] + str(v[1])
                else:
                    n = 0
                    figname += '#'
                    for col_name, cond in zip(k, v):
                        if n > 0:
                            figname += '_or_'
                        if type(cond[1]) == str:
                            figname += '(self.df.' + col_name + cond[0] + '\'' + str(cond[1]) + '\'' + ')'
                        else:
                            figname += '(self.df.' + col_name + cond[0] + str(cond[1]) + ')'
                        n += 1
            return figname

        savename = gen_fig_name(dict_cond_cols)

        # create plot
        result = df.pivot(index=x, columns=y, values=value)

        seaborn_plot = sns.heatmap(result, annot=True, fmt=".3f", cmap='viridis', ax=ax)

        axes = seaborn_plot.axes
        if ylim is not None and ystep is not None:
            axes.set_ylim(ylim[0], ylim[1])
            axes.set_yticks(np.arange(ylim[0], ylim[1], (ylim[1] - ylim[0]) / ystep))
        axes.set_title(savename)

        if figname is not None:
            savename = figname
        # plt.show()
        # fig.savefig(self.plot_dir + savename + '.png', format='png', bbox_inches='tight')

    def collect_data(self):
        print(":::Collecting log data from: %s" % (self.data_dir))
        # Loop over folder
        all_file_paths = [y for x in os.walk(self.data_dir) for y in glob(os.path.join(x[0], '*.csv'))]
        all_file_paths = sorted(all_file_paths)
        print(all_file_paths)

        # Get Column Headers
        example_path = all_file_paths[0]
        if platform.system() == 'Windows':
            elem_path = example_path.split('\\')
        else:
            elem_path = example_path.split('/')
        file_name = elem_path[-1].split('.csv')[0]

        columns = file_name.split('_')[0::2]

        # Add Arch Columns
        col_net_arch = 'net_arch'
        columns.extend([col_net_arch])

        # Add Data Columns
        df = pd.read_csv(example_path)
        columns.extend(list(df))

        # Collect Data
        summary_list = []
        for idx, file_path in enumerate(all_file_paths):
            if platform.system() == 'Windows':
                elem_path = file_path.split('\\')
            else:
                elem_path = file_path.split('/')
            file_name = elem_path[-1].split('.csv')[0]

            # Get Values of Columns
            columns_val = file_name.split('_')[1::2]

            # Add Arch Columns
            str_IN = str(columns_val[columns.index('IN')])
            # str_L = str(columns_val[columns.index('L')])
            str_H = str(columns_val[columns.index('H')])
            str_CLA = str(columns_val[columns.index('CLA')])
            # col_net_arch_val = str_IN + 'IN-' + str_L + 'L-' + str_H + 'H-' + str_CLA
            col_net_arch_val = str_IN + 'IN-' + 'L-' + str_H + 'H-' + str_CLA
            columns_val.extend([col_net_arch_val])

            # Add Data Columns
            df = pd.read_csv(file_path)
            columns_val.extend(df.loc[0, :])

            # Add a row to the description file
            row = {}
            for header, value in zip(columns, columns_val):
                if type(value) == str:
                    try:
                        row[header] = float(value)
                    except ValueError:
                        if value.isdecimal():
                            row[header] = int(value)
                        else:
                            row[header] = value
                else:
                    row[header] = value
            summary_list.append(row)

        self.df = pd.DataFrame(summary_list, columns=columns)

        # Sort Columns
        self.df = self.df.sort_values('SD', ascending=True)
        description_file = os.path.join(self.plot_dir, self.name + '_summary.csv')
        self.df.to_csv(description_file, index=False)
