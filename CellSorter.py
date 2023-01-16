from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

from scipy.signal import butter,filtfilt

import argparse

def make_s2p_dict(parent_folder_path):
    """Creates a dict of relevant suite2p outputs (ops, stat, iscell, F and F_neu files)
    Taken from aj_2p_functions"""
    fol = parent_folder_path

    stat_path = fol/'stat.npy'
    iscell_path = fol/'iscell.npy'
    op_path = fol/'ops.npy'
    f_path = fol/'F.npy'
    f_neu_path = fol/'Fneu.npy'


    ops = np.load(op_path, allow_pickle=True).item()
    stat = np.load(stat_path, allow_pickle=True)
    iscell = np.load(iscell_path, allow_pickle=True)[:,0].astype(bool)
    f = np.load(f_path, allow_pickle=True)
    f_neu = np.load(f_neu_path, allow_pickle=True)

    info_dict = {'ops':ops, 'stat':stat, 'iscell':iscell, 'F':f, 'Fneu':f_neu}
    return info_dict


def butter_lowpass_filter(data, cutoff_freq, framerate, order):
    # Taken from https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
    nyq = 0.5*framerate
    normal_cutoff = cutoff_freq / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def stats_dicts_to_3d_array(stat_list, ops):
    """This function recreates suite2p.ROI.stats_dicts_to_3d_array with a few simplifications.
    Currently, an issue with /detection/stats.py means the above suite2p function doesn't work normally. 
    The issue is solved here https://github.com/MouseLand/suite2p/pull/833/files# but for simplicity of installation, I've just remade the function"""
    Ly = ops['Ly']
    Lx = ops['Lx']

    arr_stack = np.zeros([len(stat_list), Ly, Lx])
    for ix, s in enumerate(stat_list):
        arr = np.zeros([Ly, Lx], dtype=float)
        arr[s['ypix'], s['xpix']] = 1
        arr_stack[ix] = arr

    return arr_stack



class CellSorter:
    def __init__(self, s2p_path, use_existing_sort_file = True):
        # Importing suite2p data
        if type(s2p_path) == str:
            s2p_path = Path(s2p_path)
        self.s2p_path = s2p_path

        self.s2p_dict = make_s2p_dict(self.s2p_path)
        self.ops, self.stat, self.iscell, self.F, self.Fneu = self.s2p_dict.values()
        self.footprints = stats_dicts_to_3d_array(self.stat, self.ops)
        self.num_ROIs = len(self.footprints)

        # Create array for holding sorted values. 0 = unsorted, -1 = not cell, 1 = cell
        if use_existing_sort_file:
            # Look for existing sort file
            existing_sort_path = self.s2p_path/'cell_sort_scores.npy'
            if existing_sort_path.is_file():
                self.sort_status = np.load(existing_sort_path)
                print(f'Loaded previous cell_sort file at {existing_sort_path}')
            else:
                print('No previous cell_sort file found. Starting from scratch')
                self.sort_status = np.zeros(self.num_ROIs, dtype=int)
        else:
            print("Starting new cell_sort file")
            self.sort_status = np.zeros(self.num_ROIs, dtype=int)

        # This index position counter will be updated as we move through ROIs in the GUI
        self.ix_position = 0

        self.display_footprint = True  # flag for whether we will display the ROI footprint
        self.display_lowpass_trace = True  # flag for displaying smoothed ROI activity trace

    def backup_iscell(self):
        # Copies existing iscell.npy to a backup file 
        iscell_path = self.s2p_path/'iscell.npy'
        shutil.copy(iscell_path, iscell_path.with_name('iscell_backup.npy'))

    def gen_s2p_fig(self, ix, show_cell = True, show_lowpass = True, cutoff_freq=0.1, framerate=20, order=2):
        # Generates a figure of ROI footprint and track
        fig, ax = plt.subplots(2,1, figsize=(15,7), gridspec_kw={'height_ratios':[2,1]})

        ax[0].imshow(self.ops['meanImg'], cmap='gray')
        if show_cell:
            ax[0].imshow(np.ma.masked_where(self.footprints[ix] == 0, self.footprints[ix]), vmin=0, cmap='viridis', alpha=0.75)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].plot(self.F[ix], color='k', alpha=0.75)
        ax[1].plot(self.Fneu[ix], color='blue', zorder=0)
        if show_lowpass:
            ax[1].plot(butter_lowpass_filter(self.F[ix], cutoff_freq, framerate , order), color='r', alpha=0.5)
        return fig
    
    def save_sort_status(self):
        # Saves the current sort_status array to cell_sort_scores.npy
        np.save(self.s2p_path/'cell_sort_scores.npy', self.sort_status)

    def get_status_str(self):
        # Convenience function, returns a string associated with the current value in the sort_status array
        status_map = {1: 'Good', -1: 'Bad', 0: 'Unsorted'}
        return status_map[self.sort_status[self.ix_position]]

    def GUI_loop(self):

        # Create a GUI window and text
        self.root = tk.Tk()
        self.root.geometry("1500x800")
        self.root.title("Cell Sorter")

        self.text_var_ix = tk.StringVar()   # Create a variable to hold the current ix
        self.text_label_ix = tk.Label(self.root, textvariable = self.text_var_ix)
        self.text_var_ix.set(f'ROI {self.ix_position}/{self.num_ROIs}')

        self.text_var_current_score = tk.StringVar()
        self.text_var_current_score.set(f'Current cell: {self.get_status_str()}')
        self.text_label_current_score = tk.Label(self.root, textvariable = self.text_var_current_score)

        self.text_var_remaining = tk.StringVar()
        self.text_var_remaining.set(f'Unsorted cells remaining: {len(self.sort_status[self.sort_status == 0])}')
        self.text_label_remaining = tk.Label(self.root, textvariable = self.text_var_remaining)

        # Create a canvas to display the figure
        self.canvas = FigureCanvasTkAgg(self.gen_s2p_fig(0), master=self.root)
        self.canvas.draw()

        # Lay out elements in the GUI window
        r = 0
        self.canvas.get_tk_widget().grid(row=r, column=0, columnspan=10, sticky='nsew')

        r = 1
        self.text_label_ix.grid(row=r, column=4, sticky='nsew')
        self.text_label_current_score.grid(row=r, column=5, sticky='nsew')
        self.text_label_remaining.grid(row=r, column=6, sticky='nsew')
        self.button_toggle_footprint = tk.Button(self.root, text = 'Toggle footprint display (f)', command = self._toggle_footprint)
        self.button_toggle_footprint.grid(row=r, column=7, sticky='ns')

        r = 2
        self.button_sort_bad = tk.Button(self.root, text='Bad (s)', command = self._sort_cell_bad)
        self.button_sort_bad.grid(row=r, column=5, sticky='nsew')
        self.button_sort_good = tk.Button(self.root, text='Good (w)', command = self._sort_cell_good)
        self.button_sort_good.grid(row=r, column=6, sticky='nsew')
        self.button_toggle_lowpass_trace = tk.Button(self.root, text = 'Toggle trace lowpass filter (t)', command = self._toggle_lowpass_trace)
        self.button_toggle_lowpass_trace.grid(row=r, column=7, sticky='ns')

        r = 3
        self.button_previous = tk.Button(self.root, text="Previous (a)", command=self._previous_ROI)
        self.button_previous.grid(row = r, column = 0, columnspan=2, sticky='nsew')
        self.button_next = tk.Button(self.root, text="Next (d)", command=self._next_ROI)
        self.button_next.grid(row = r, column = 8, columnspan=2, sticky='nsew')
        self.button_sort_all_bad = tk.Button(self.root, text="Sort all remaining cells as bad", command=self._sort_remaining_warning_message)
        self.button_sort_all_bad.grid(row = r, column = 4)

        self.root.bind("w", self._sort_cell_good)
        self.root.bind("s", self._sort_cell_bad)
        self.root.bind("a", self._previous_ROI)
        self.root.bind("d", self._next_ROI)
        self.root.bind("f", self._toggle_footprint)
        self.root.bind("t", self._toggle_lowpass_trace)


        num_columns = 10
        num_rows = 4
        for i in range(num_columns):
            self.root.columnconfigure(i, weight=1)
        for i in range(num_rows):
            self.root.rowconfigure(i, weight=1)

        self.root.rowconfigure(0, weight=2)

        self.root.mainloop()

    def _next_ROI(self, event=None):
        # Set new position
        self.ix_position += 1
        if self.ix_position > (self.num_ROIs - 1):
            self.ix_position = 0
        # Reset text
        self.text_var_ix.set(f'ROI {self.ix_position}/{self.num_ROIs}')
        self.text_var_current_score.set(f'Current cell: {self.get_status_str()}')
        # Reset and draw display
        self.display_footprint = True
        self.display_lowpass_trace = True
        self.canvas.figure = self.gen_s2p_fig(self.ix_position, self.display_footprint, self.display_lowpass_trace)
        self.canvas.draw()
        plt.close(self.canvas.figure)

    def _previous_ROI(self, event=None):
        # Set new position
        self.ix_position -= 1
        if self.ix_position == -1:
            self.ix_position = self.num_ROIs-1
        # Reset text
        self.text_var_ix.set(f'ROI {self.ix_position}/{self.num_ROIs}')
        self.text_var_current_score.set(f'Current cell: {self.get_status_str()}')
        # Reset and draw display
        self.display_footprint = True
        self.display_lowpass_trace = True
        self.canvas.figure = self.gen_s2p_fig(self.ix_position, self.display_footprint, self.display_lowpass_trace)
        self.canvas.draw()
        plt.close(self.canvas.figure)

    def _sort_cell_good(self, event=None):
        # Labels current position as good
        self.sort_status[self.ix_position] = 1
        self.text_var_current_score.set(f'Current cell: {self.get_status_str()}')
        self.text_var_remaining.set(f'Unsorted cells remaining: {len(self.sort_status[self.sort_status == 0])}')
        self.save_sort_status()

    def _sort_cell_bad(self, event=None):
        # Labels current position as bad
        self.sort_status[self.ix_position] = -1
        self.text_var_current_score.set(f'Current cell: {self.get_status_str()}')
        self.text_var_remaining.set(f'Unsorted cells remaining: {len(self.sort_status[self.sort_status == 0])}')
        self.save_sort_status()

    def _sort_remaining_bad(self):
        # Labels all remaining unsorted positions as bad
        self.sort_status[self.sort_status == 0] = -1
        self.save_sort_status()
        self.text_var_remaining.set(f'Unsorted cells remaining: {len(self.sort_status[self.sort_status == 0])}')
        self.text_var_current_score.set(f'Current cell: {self.get_status_str()}')

    def _sort_remaining_warning_message(self):
        # before proceeding to sort all cells as bad, ask user if they are sure
        result = tk.messagebox.askquestion("Confirmation", "Are you sure you want to sort all remaining cells as bad?", icon='warning')
        if result == 'yes':
            # proceed with the action
            self._sort_remaining_bad()
        else:
            # cancel the action
            pass

    def _toggle_footprint(self, event=None):
        # Toggles display of the ROI footprint
        self.display_footprint = not self.display_footprint # toggle the value
        self.canvas.figure = self.gen_s2p_fig(self.ix_position, self.display_footprint, self.display_lowpass_trace)
        self.canvas.draw()

    def _toggle_lowpass_trace(self, event=None):
        # Toggles display of the ROI footprint
        self.display_lowpass_trace = not self.display_lowpass_trace # toggle the value
        self.canvas.figure = self.gen_s2p_fig(self.ix_position, self.display_footprint, self.display_lowpass_trace)
        self.canvas.draw()




if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("s2p_path", help="Path to the output folder generated by suite2p")
    #args = parser.parse_args()
    #p = Path(args.s2p_path)

    p = Path(r"F:\round_2\684169\ms3\tone_d1\suite2p\plane0")
    sorter = CellSorter(p)

    sorter.GUI_loop()