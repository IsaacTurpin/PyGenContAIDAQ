import csv
import os
import nidaqmx
import numpy as np
from nidaqmx import constants
from nidaqmx.constants import TerminalConfiguration
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
matplotlib.use("TkAgg")
from datetime import datetime


class VoltageContinuousInput(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.master.title("Voltage - Continuous Input")
        self.master.geometry("1100x700")
        self.run = False
        self.task = None
        self.create_widgets()
        self.pack()

    def create_widgets(self):
        # Main frames for the GUI
        self.channelSettingsFrame = ChannelSettings(self, title="Channel Settings")
        self.channelSettingsFrame.grid(row=0, column=1, sticky="ew", pady=(20, 0), padx=(20, 20), ipady=10)

        self.inputSettingsFrame = InputSettings(self, title="Input Settings")
        self.inputSettingsFrame.grid(row=1, column=1, pady=(20, 0), padx=(20, 20), ipady=10)
        self.inputSettingsFrame.stopButton['state'] = 'disabled'

        # Pass self (parent) to GraphData
        self.graphDataFrame = GraphData(self)
        self.graphDataFrame.grid(row=0, rowspan=2, column=2, pady=(20, 0), ipady=10)

        self.fileSave = FileSaving(self)

    def start_task(self):
        try:
            # Validate inputs
            physical_channel = self.channelSettingsFrame.physicalChannelEntry.get()
            sample_rate = self.inputSettingsFrame.sampleRateEntry.get() # how many data points are collected per second
            scale = self.channelSettingsFrame.scale_entry.get()
            offset = self.channelSettingsFrame.offset_entry.get()

            if not physical_channel or not sample_rate or not scale or not offset:
                messagebox.showerror("Input Error", "All fields are required!")
                return

            # Disable start button and enable stop button
            self.inputSettingsFrame.startButton['state'] = 'disabled'
            self.inputSettingsFrame.stopButton['state'] = 'enabled'
            self.continueRunning = True

            # Parse channel settings
            file_data_title = self.parse_channel_titles(physical_channel)
            self.no_chl = len(file_data_title)
            self.graphDataFrame.define_xy_array(self.no_chl, file_data_title)

            # Parse scale and offset
            self.scale = [float(s) for s in scale.split(',')]
            self.offset = [float(o) for o in offset.split(',')]


            # Configure DAQ task
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(physical_channel, min_val=-10, max_val=10,
                                                      terminal_config=TerminalConfiguration.RSE)
            self.task.timing.cfg_samp_clk_timing(float(sample_rate),
                                                 sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            self.task.start()

            # Initialize file saving if enabled
            if self.fileSave.save_data.get():
                self.fileSave.get_data_title(file_data_title, sample_rate, self.scale, self.offset)

            # Start data acquisition loop
            self.no_time = -1
            self.master.after(10, self.run_task)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.stop_task()

    def run_task(self):
        try:
            if self.continueRunning:
                samples_available = self.task._in_stream.avail_samp_per_chan
                if samples_available > 0:
                    # Calculate elapsed time based on the number of samples and sample rate
                    sample_rate = float(self.inputSettingsFrame.sampleRateEntry.get())
                    elapsed_time = samples_available / sample_rate  # Time for the current batch of samples

                    # Update the total elapsed time
                    if not hasattr(self, 'total_elapsed_time'):
                        self.total_elapsed_time = 0  # Initialize total elapsed time
                    self.total_elapsed_time += elapsed_time

                    # Generate time axis for the current batch of samples
                    time_axis = np.linspace(
                        self.total_elapsed_time - elapsed_time,  # Start time for this batch
                        self.total_elapsed_time,  # End time for this batch
                        samples_available,  # Number of samples
                        endpoint=False  # Ensure the last point is not included
                    )

                    # Read data from the DAQ
                    vals = self.task.read(samples_available)

                    # Save data to file if enabled
                    if self.fileSave.save_data.get():
                        self.fileSave.write_to_file(self.total_elapsed_time, self.no_chl, vals)

                    # Plot data
                    self.graphDataFrame.plot_data(self.no_chl, samples_available, time_axis, self.scale, self.offset,
                                                  vals)

                # Continue the task
                self.master.after(10, self.run_task)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.stop_task()

    def stop_task(self):
        self.continueRunning = False
        if self.task:
            self.task.stop()
            self.task.close()
        self.inputSettingsFrame.startButton['state'] = 'enabled'
        self.inputSettingsFrame.stopButton['state'] = 'disabled'
        if self.fileSave.save_data.get():
            self.fileSave.close_file()

    def parse_channel_titles(self, physical_channel):
        chl_name = 'ai'
        file_data_title = []
        if ',' in physical_channel:
            split_chl = physical_channel.split(',')
            for item in split_chl:
                file_data_title.append(f"Channel {item.partition(chl_name)[2]}")
        elif ':' in physical_channel:
            chl_range = physical_channel.partition(chl_name)[2].split(':')
            for i in range(int(chl_range[0]), int(chl_range[1]) + 1):
                file_data_title.append(f"Channel {i}")
        else:
            file_data_title.append(f"Channel {physical_channel.partition(chl_name)[2]}")
        return file_data_title


class ChannelSettings(tk.LabelFrame):
    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.grid_columnconfigure(0, weight=1)
        self.xPadding = (30, 30)
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Physical Channel \n(e.g., Dev1/ai0, Dev1/ai0:3, Dev1/ai0,Dev1/ai2)").grid(row=0, sticky='w', padx=self.xPadding, pady=(10, 0))
        self.physicalChannelEntry = ttk.Entry(self)
        self.physicalChannelEntry.insert(0, "Dev1/ai0:1")
        self.physicalChannelEntry.grid(row=1, sticky="ew", padx=self.xPadding)

        ttk.Label(self, text="Scale (comma-separated, e.g., 1, 2)").grid(row=2, sticky='w', padx=self.xPadding, pady=(10, 0))
        self.scale_entry = ttk.Entry(self)
        self.scale_entry.insert(0, "1, 1")
        self.scale_entry.grid(row=3, sticky="ew", padx=self.xPadding)

        ttk.Label(self, text="Offset (comma-separated, e.g., 0, 0)").grid(row=4, sticky='w', padx=self.xPadding, pady=(10, 0))
        self.offset_entry = ttk.Entry(self)
        self.offset_entry.insert(0, "0, 0")
        self.offset_entry.grid(row=5, sticky="ew", padx=self.xPadding, pady=(0, 10))

        ttk.Label(self, text="Calibrated Data = Raw Data * Scale + Offset").grid(row=6, sticky="w", padx=self.xPadding, pady=(10, 0))


class InputSettings(tk.LabelFrame):
    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.xPadding = (30, 30)
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Sample Rate").grid(row=0, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10, 0))
        self.sampleRateEntry = ttk.Entry(self)
        self.sampleRateEntry.insert(0, "1000")
        self.sampleRateEntry.grid(row=1, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.startButton = ttk.Button(self, text="Start Task", command=self.parent.start_task)
        self.startButton.grid(row=4, column=0, sticky='w', padx=self.xPadding, pady=(10, 0))

        self.stopButton = ttk.Button(self, text="Stop Task", command=self.parent.stop_task)
        self.stopButton.grid(row=4, column=1, sticky='e', padx=self.xPadding, pady=(10, 0))


class GraphData(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.graph_title = ttk.Label(self, text="Voltage Input")
        self.parent = parent  # Store the parent reference
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Calibrated Data")
        self.ax.set_ylim(-10, 10)  # Set y-axis limits
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.get_tk_widget().pack()
        self.xdata = np.array([]) # Stores the time values (x-axis)
        self.ydata = [] # Stores the voltage values (y-axis)
        self.display_duration = 10  # Display last n seconds of data
        self.min_value = np.inf  # Initialize min_value to positive infinity
        self.max_value = -np.inf  # Initialize max_value to negative infinity

    def define_xy_array(self, array_no, data_title):
        self.legend_title = data_title
        self.xdata = np.array([])  # x-axis data buffer
        self.ydata = [[] for _ in range(array_no)]  # y-axis data buffer

    def plot_data(self, chl_number, sample_number, x_vals, scale, offset, y_vals):
        self.ax.clear()  # Clear previous data without losing axis settings

        # Calculate the number of samples to display for n seconds
        sample_rate = float(self.parent.inputSettingsFrame.sampleRateEntry.get())
        display_samples = int(sample_rate * self.display_duration)

        # Update data buffers
        if len(self.xdata) < display_samples:
            self.xdata = np.append(self.xdata, x_vals)
            if chl_number == 1:
                self.ydata = np.append(self.ydata, y_vals)
            else:
                for i in range(chl_number):
                    self.ydata[i] = np.append(self.ydata[i], y_vals[i])
        else:
            # Remove the oldest data to make room for new data
            self.xdata = np.roll(self.xdata, -sample_number)
            self.xdata[-sample_number:] = x_vals
            if chl_number == 1:
                self.ydata = np.roll(self.ydata, -sample_number)
                self.ydata[-sample_number:] = y_vals
            else:
                for i in range(chl_number):
                    self.ydata[i] = np.roll(self.ydata[i], -sample_number)
                    self.ydata[i][-sample_number:] = y_vals[i]

        # Trim the data to only the last n seconds
        if len(self.xdata) > display_samples:
            self.xdata = self.xdata[-display_samples:]
            if chl_number == 1:
                self.ydata = self.ydata[-display_samples:]
            else:
                for i in range(chl_number):
                    self.ydata[i] = self.ydata[i][-display_samples:]

        # Apply scaling and offset to the data
        calibrated_data = []
        for i in range(chl_number):
            if chl_number == 1:
                calibrated_data = np.array(self.ydata) * float(scale[i]) + float(offset[i])
            else:
                calibrated_data.append(np.array(self.ydata[i]) * float(scale[i]) + float(offset[i]))

        # Find the minimum and maximum values of the calibrated data
        if chl_number == 1:
            current_min = np.min(calibrated_data)
            current_max = np.max(calibrated_data)
        else:
            current_min = min(np.min(channel) for channel in calibrated_data)
            current_max = max(np.max(channel) for channel in calibrated_data)

        # Update the overall min and max values
        if current_min < self.min_value:
            self.min_value = current_min
        if current_max > self.max_value:
            self.max_value = current_max

        # Add a margin to the min and max values
        margin = 0.1 * (self.max_value - self.min_value)  # 10% of the data range
        y_min = self.min_value - margin
        y_max = self.max_value + margin

        # Set y-axis limits with the added margin
        self.ax.set_ylim(y_min, y_max)

        # Plot the data
        if chl_number == 1:
            self.ax.plot(self.xdata, calibrated_data, label=self.legend_title[0])
        else:
            for i in range(chl_number):
                self.ax.plot(self.xdata, calibrated_data[i], label=self.legend_title[i])

        # Set x-axis limits to show the last n seconds
        if len(self.xdata) > 0:
            self.ax.set_xlim(max(0, self.xdata[-1] - self.display_duration), self.xdata[-1])

        # Add legend
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True)
        self.graph.draw()


class FileSaving:
    def __init__(self, parent):
        self.parent = parent
        self.file = None
        self.writer = None
        self.var_dict = {}
        self.save_data = tk.BooleanVar(value=False)
        self.folder_path = tk.StringVar(value="Results")
        self.create_widgets()

    def create_widgets(self):
        # Checkbox to enable/disable data saving
        ttk.Checkbutton(self.parent, text="Save Data to File", variable=self.save_data).grid(row=2, column=1, sticky='w', padx=(20, 20), pady=(10, 0))

        # Folder path selection
        ttk.Label(self.parent, text="Save Folder:").grid(row=3, column=1, sticky='w', padx=(20, 20), pady=(10, 0))
        self.folderEntry = ttk.Entry(self.parent, textvariable=self.folder_path)
        self.folderEntry.grid(row=4, column=1, sticky='ew', padx=(20, 20))
        ttk.Button(self.parent, text="Browse", command=self.browse_folder).grid(row=4, column=1, sticky='e', padx=(20, 20))

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path.set(folder_selected)

    def get_data_title(self, file_title, sample_rate, data_scale, data_offset):
        # Create folder if it doesn't exist
        if not os.path.exists(self.folder_path.get()):
            os.makedirs(self.folder_path.get())

        # Generate file name
        self.file_name = os.path.join(self.folder_path.get(), f"Result_{datetime.today().strftime('%d%b%Y_%Hh%Mm%Ss')}.csv")

        # Initialize data dictionary
        self.var_dict = {ft: [] for ft in file_title}
        self.var_dict['Time (s)'] = []

        # Open file and write metadata
        self.file = open(self.file_name, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['Sample Rate', str(sample_rate)])
        self.writer.writerow(['Scale'] + [str(s) for s in data_scale])
        self.writer.writerow(['Offset'] + [str(o) for o in data_offset])
        self.writer.writerow(self.var_dict.keys())

        self.delta_t = 1 / float(sample_rate)

    def write_to_file(self, time_start, data_list_no, saving_data):
        time_col = np.arange(time_start - 1, time_start, self.delta_t)  # Use total_elapsed_time
        self.var_dict['Time (s)'] = time_col
        for i, key in enumerate(self.var_dict.keys()):
            if key != 'Time (s)':
                self.var_dict[key] = saving_data[i] if data_list_no > 1 else saving_data
        self.writer.writerows(zip(*self.var_dict.values()))
        return time_col

    def close_file(self):
        if self.file:
            self.file.close()


# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = VoltageContinuousInput(root)
    app.mainloop()
