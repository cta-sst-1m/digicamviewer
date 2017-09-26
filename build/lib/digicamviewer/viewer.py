import numpy as np
import sys
from ctapipe import visualization
from . import geometry
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.widgets import Button, RadioButtons, CheckButtons
from itertools import cycle
from matplotlib.colors import LogNorm
import matplotlib as mpl
import scipy.stats
import matplotlib.animation as animation
from cts_core import camera


class EventViewer2():

    def __init__(self, event_stream, camera_config_file, n_samples, scale='lin', limits_colormap=None, limits_readout=None, time_bin_start=0, pixel_id_start=0, telescope_id_start=1):

        mpl.figure.autolayout = False
        self.first_call = True
        self.event_stream = event_stream
        self.scale = scale
        self.limits_colormap = limits_colormap if limits_colormap is not None else [-np.inf, np.inf]
        self.limits_readout = limits_readout
        self.time_bin = time_bin_start
        self.pixel_id = pixel_id_start
        self.telescope_id = telescope_id_start
        self.mask_pixels = False
        self.hillas = False

        self.event_id = None
        self.r0_container = None
        self.r1_container = None
        self.dl0_container = None
        self.dl1_container = None
        self.dl2_container = None
        self.trigger_output = None
        self.trigger_input = None
        self.trigger_patch = None
        self.nsb = np.nan
        self.gain_drop = np.nan
        self.baseline = np.nan
        self.std = np.nan
        self.flag = None

        self.event_clicked_on = Event_Clicked(pixel_start=self.pixel_id)
        self.camera = camera.Camera(_config_file=camera_config_file)
        self.geometry = geometry.generate_geometry(camera=self.camera)[0]
        self.n_pixels = len(self.camera.Pixels)
        self.n_samples = n_samples

        self.readout_view_types = ['raw', 'baseline substracted', 'photon', 'trigger input', 'trigger output', 'patch', 'reconstructed charge']
        self.readout_view_type = 'raw'

        self.camera_view_types = ['sum', 'std', 'mean', 'max', 'time']
        self.camera_view_type = 'std'

        self.figure = plt.figure(figsize=(20, 10))
        self.axis_readout = self.figure.add_subplot(122)
        self.axis_camera = self.figure.add_subplot(121)
        self.axis_camera.axis('off')

        self.axis_readout.set_xlabel('t [ns]')
        self.axis_readout.set_ylabel('[ADC]')
        self.axis_readout.legend(loc='upper right')
        self.axis_readout.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        self.axis_readout.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

        self.trace_readout, = self.axis_readout.step(np.arange(self.n_samples) * 4, np.ones(self.n_samples), where='mid')
        self.trace_time_plot, = self.axis_readout.plot(np.array([self.time_bin, self.time_bin]) * 4, np.ones(2), color='k',
                                                       linestyle='--')

        self.camera_visu = visualization.CameraDisplay(self.geometry, ax=self.axis_camera, title='', norm=self.scale,
                                                       cmap='viridis',
                                                       allow_pick=True)
        if limits_colormap is not None:
            self.camera_visu.set_limits_minmax(limits_colormap[0], limits_colormap[1])

        self.camera_visu.image = np.zeros(self.n_pixels)
        self.camera_visu.cmap.set_bad(color='k')
        self.camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05, shrink=.85)

        if self.scale == 'log':
            self.camera_visu.colorbar.set_norm(LogNorm(vmin=1, vmax=None, clip=False))
        self.camera_visu.colorbar.set_label('[LSB]')
        self.camera_visu.axes.get_xaxis().set_visible(False)
        self.camera_visu.axes.get_yaxis().set_visible(False)
        self.camera_visu.on_pixel_clicked = self.draw_readout

        # Buttons

        self.axis_next_event_button = self.figure.add_axes([0.35, 0.9, 0.15, 0.07], zorder=np.inf)
        self.axis_next_camera_view_button = self.figure.add_axes([0., 0.85, 0.1, 0.15], zorder=np.inf)
        self.axis_next_view_type_button = self.figure.add_axes([0., 0.18, 0.1, 0.15], zorder=np.inf)
        self.axis_check_button = self.figure.add_axes([0.35, 0.18, 0.1, 0.1], zorder=np.inf)
        self.axis_next_camera_view_button.axis('off')
        self.axis_next_view_type_button.axis('off')
        self.button_next_event = Button(self.axis_next_event_button, 'Next')
        self.radio_button_next_camera_view = RadioButtons(self.axis_next_camera_view_button, self.camera_view_types,
                                                          active=self.camera_view_types.index(self.camera_view_type))
        self.radio_button_next_view_type = RadioButtons(self.axis_next_view_type_button, self.readout_view_types,
                                                        active=self.readout_view_types.index(self.readout_view_type))

        self.check_button = CheckButtons(self.axis_check_button, ('mask', 'hillas'), (self.mask_pixels, self.hillas))
        self.radio_button_next_view_type.set_active(self.readout_view_types.index(self.readout_view_type))
        self.radio_button_next_camera_view.set_active(self.camera_view_types.index(self.camera_view_type))

    #def __next__(self):

    #    self.next(step=1)

    #    return self.event_stream

    def next(self, event=None, step=1):

        for i, event_iterator in zip(range(step), self.event_stream):

            pass

        self.event_id = event_iterator.r0.event_id
        self.r0_container = event_iterator.r0.tel[self.telescope_id]
        self.r1_container = event_iterator.r1.tel[self.telescope_id]
        self.dl0_container = event_iterator.dl0.tel[self.telescope_id]
        self.dl1_container = event_iterator.dl1.tel[self.telescope_id]
        self.dl2_container = event_iterator.dl2
        zero_image = np.zeros((self.n_pixels, self.n_samples))
        self.baseline = self.r0_container.baseline if self.r0_container.baseline is not None else np.nan * zero_image
        self.std = self.r0_container.standard_deviation if self.r0_container.standard_deviation is not None else np.nan * zero_image
        self.flag = self.r0_container.flag if self.r0_container.flag is not None else np.nan * zero_image
        self.nsb = self.r1_container.nsb if self.r1_container.adc_samples is not None else np.nan * zero_image
        self.gain_drop = self.r1_container.gain_drop if self.r1_container.adc_samples is not None else np.nan * zero_image

        if self.first_call:

            self.first_call = False

        self.update()

    def update(self):

        self.draw_readout(self.pixel_id)
        self.draw_camera()
        self.button_next_event.label.set_text('Next : current event #%d' % (self.event_id))

    def draw(self):

        self.next()
        self.button_next_event.on_clicked(self.next)
        self.radio_button_next_camera_view.on_clicked(self.next_camera_view)
        self.radio_button_next_view_type.on_clicked(self.next_view_type)
        self.check_button.on_clicked(self.draw_on_camera)
        self.figure.canvas.mpl_connect('key_press_event', self.press)
        self.camera_visu._on_pick(self.event_clicked_on)
        plt.show()

    def draw_camera(self, plot_hillas=False):

        self.camera_visu.image = self.compute_image()

        if plot_hillas:
            self.camera_visu.overlay_moments(self.dl2_container.shower)

    def draw_readout(self, pixel):

        y = self.compute_trace()[pixel]
        limits_y = self.limits_readout if self.limits_readout is not None else [np.min(y), np.max(y) + 10]
        self.pixel_id = pixel
        self.event_clicked_on.ind[-1] = self.pixel_id
        self.trace_readout.set_ydata(y)
        self.trace_readout.set_label(
            '%s : %d, bin : %d \n Flag = %0.1f \n $B= %0.2f$ [LSB] \n $\sigma = %0.2f$ [LSB]'
            ' \n $f_{nsb} = %0.2f$ [GHz] \n $G_{drop}= %0.2f$'
            % (self.readout_view_type, self.pixel_id, self.time_bin, self.flag,
               self.baseline[self.pixel_id], self.std[self.pixel_id], self.nsb[self.pixel_id], self.gain_drop[self.pixel_id]))
        self.trace_time_plot.set_ydata(limits_y)
        self.trace_time_plot.set_xdata(self.time_bin * 4)
        self.axis_readout.set_ylim(limits_y)
        self.axis_readout.legend(loc='upper right')

        if self.readout_view_type in ['photon', 'reconstructed charge']:

            self.axis_readout.set_ylabel('[p.e.]')

        else:

            self.axis_readout.set_ylabel('[LSB]')

    def compute_trace(self):

        if self.readout_view_type in self.readout_view_types:

            if self.readout_view_type == 'raw':

                image = np.array(list(self.r0_container.adc_samples.values()))

            elif self.readout_view_type == 'trigger output' and self.trigger_output is not None:

                # image = np.array([self.trigger_output[pixel.patch] for pixel in self.camera.Pixels])
                image = np.zeros((self.n_pixels, self.n_samples))
                print('%s not implemented' % self.readout_view_type)

            elif self.readout_view_type == 'trigger input' and self.trigger_input is not None:

                image = np.zeros((self.n_pixels, self.n_samples))
                print('%s not implemented' % self.readout_view_type)

            elif self.readout_view_type == 'patch' and self.trigger_patch is not None:

                image = np.zeros((self.n_pixels, self.n_samples))
                print('%s not implemented' % self.readout_view_type)

            elif self.readout_view_type == 'photon' and self.dl1_container.pe_samples_trace is not None:

                image = self.dl1_container.pe_samples_trace

            elif self.readout_view_type == 'baseline substracted' and self.r1_container.adc_samples is not None:


                image = np.array(list(self.r1_container.adc_samples.values()))

            elif self.readout_view_type == 'reconstructed charge' and (self.dl1_container.time_bin is not None or self.dl1_container.pe_samples is not None):

                image = np.zeros((self.n_pixels, self.n_samples))
                time_bins = self.dl1_container.time_bin
                image[time_bins] = self.dl1_container.pe_samples

            else:

                image = np.zeros((self.n_pixels, self.n_samples))

        return image

    def next_camera_view(self, camera_view, event=None):

        self.camera_view_type = camera_view
        if self.readout_view_type in ['photon','reconstructed charge']:
            self.camera_visu.colorbar.set_label('[p.e.]')

        else:
            self.camera_visu.colorbar.set_label('[LSB]')

        self.update()

    def next_view_type(self, view_type, event=None):

        self.readout_view_type = view_type

        if view_type in ['photon', 'reconstructed charge']:
            self.camera_visu.colorbar.set_label('[p.e.]')

        else:
            self.camera_visu.colorbar.set_label('[LSB]')




        self.update()

    def draw_on_camera(self, to_draw_on, event=None):

        if to_draw_on == 'hillas':

            if self.hillas:

                self.hillas = False

            else:

                self.hillas = True

        if to_draw_on == 'mask':

            if self.mask_pixels:

                self.mask_pixels = False

            else:

                self.mask_pixels = True

        self.update()

    def set_time(self, time):

        if time < self.n_samples and time >= 0:

            self.time_bin = time
            self.update()

    def set_pixel(self, pixel_id):

        if pixel_id < self.n_samples and pixel_id >= 0:
            self.pixel_id = pixel_id
            self.update()

    def compute_image(self):

        image = self.compute_trace()

        if self.camera_view_type in self.camera_view_types:

            if self.camera_view_type == 'mean':

                self.image = np.mean(image, axis=1)

            elif self.camera_view_type == 'std':

                self.image = np.std(image, axis=1)

            elif self.camera_view_type == 'max':

                self.image = np.max(image, axis=1)

            elif self.camera_view_type == 'sum':

                self.image = np.sum(image, axis=1)

            elif self.camera_view_type == 'time':

                self.image = image[:, self.time_bin]

        else:

            print('Cannot compute for camera type : %s' % self.camera_view)

        if self.limits_colormap is not None:

            mask = (self.image >= self.limits_colormap[0]) * (self.image <= self.limits_colormap[1])

        # if self.mask_pixels:

            #mask = mask * self.dl1_container.cleaning_mask
            image[self.dl1_container.cleaning_mask] = 0


        if self.hillas:

            self.camera_visu.overlay_moments(self.dl2_container.shower)

        return np.ma.masked_where(~mask, self.image)

    def press(self, event):

        sys.stdout.flush()

        if event.key == 'enter':
            self.next()

        if event.key == 'right':
            self.set_time(self.time_bin + 1)

        if event.key == 'left':
            self.set_time(self.time_bin - 1)

        if event.key == '+':
            self.set_pixel(self.pixel_id + 1)

        if event.key == '-':
            self.set_pixel(self.pixel_id - 1)

        if event.key == 'h':
            self.axis_next_event_button.set_visible(False)

        if event.key == 'v':
            self.axis_next_event_button.set_visible(True)

        self.update()



class EventViewer():

    def __init__(self, event_stream, camera_config_file, scale='log', limits_colormap=None, bin_start=0,
                 threshold=0, pixel_start=0, view_type='pixel', camera_view='std', limits_readout=None, telescope_id=1):


        self.first_call = True
        mpl.figure.autolayout = False
        self.scale = scale
        self.colorbar_limits = limits_colormap if limits_colormap is not None else [0, np.inf]
        self.event_iterator = event_stream

        self.time = bin_start
        self.pixel_id = pixel_start

        self.telescope_id = telescope_id
        self.data = np.array(list(self.event_iterator.__next__().r0.tel[self.telescope_id].adc_samples.values()))
        self.n_pixels = self.data.shape[0]
        self.threshold = threshold

        self.event_clicked_on = Event_Clicked(pixel_start=self.pixel_id)

        self.camera = camera.Camera(_config_file=camera_config_file)
        self.geometry = geometry.generate_geometry(camera=self.camera)[0]

        self.view_type = view_type
        self.view_types = ['pixel', 'trigger_out', 'trigger_in']
        self.iterator_view_type = cycle(self.view_types)
        self.camera_view = camera_view
        self.camera_views = ['sum', 'mean', 'max', 'std', 'time', 'stacked', 'r1', 'dl0', 'dl1', 'dl2']
        self.iterator_camera_view = cycle(self.camera_views)
        self.figure = plt.figure(figsize=(20, 10))

        ## Readout

        self.axis_readout = self.figure.add_subplot(122)
        self.axis_readout.set_xlabel('t [ns]')
        self.axis_readout.set_ylabel('[LSB]')
        self.axis_readout.legend(loc='upper right')
        self.axis_readout.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        self.axis_readout.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

        if self.threshold is not None:
            self.axis_readout.axhline(y=self.threshold, linestyle='--', color='k')  # , label='threshold')

        self.limits_readout = limits_readout

        ## Camera

        self.axis_camera = self.figure.add_subplot(121)
        self.axis_camera.axis('off')
        self.camera_visu = visualization.CameraDisplay(self.geometry, ax=self.axis_camera, title='', norm=self.scale,
                                                       cmap='viridis',
                                                       allow_pick=True)
        if limits_colormap is not None:
            self.camera_visu.set_limits_minmax(limits_colormap[0], limits_colormap[1])

        self.camera_visu.image = np.zeros(self.n_pixels)
        self.camera_visu.cmap.set_bad(color='k')
        self.camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05, shrink=.85)

        if self.scale == 'log':
            self.camera_visu.colorbar.set_norm(LogNorm(vmin=1, vmax=None, clip=False))
        self.camera_visu.axes.get_xaxis().set_visible(False)
        self.camera_visu.axes.get_yaxis().set_visible(False)
        self.camera_visu.on_pixel_clicked = self.draw_readout

        ## Buttons

        readout_position = self.axis_readout.get_position()
        self.axis_next_event_button = self.figure.add_axes([0.35, 0.9, 0.15, 0.07], zorder=np.inf)
        self.axis_next_camera_view_button = self.figure.add_axes([0., 0.85, 0.1, 0.15], zorder=np.inf)
        self.axis_next_view_type_button = self.figure.add_axes([0., 0.18, 0.1, 0.15], zorder=np.inf)
        # self.axis_slider_time = self.figure.add_axes([readout_position.x0 - 0.018, readout_position.y1 + 0.005, readout_position.x1 - readout_position.x0 + 0.005, 0.02], facecolor='lightgoldenrodyellow', zorder=np.inf)
        self.axis_next_camera_view_button.axis('off')
        self.axis_next_view_type_button.axis('off')
        self.button_next_event = Button(self.axis_next_event_button, 'Next')
        self.radio_button_next_camera_view = RadioButtons(self.axis_next_camera_view_button, self.camera_views,
                                                          active=self.camera_views.index(self.camera_view))
        self.radio_button_next_view_type = RadioButtons(self.axis_next_view_type_button, self.view_types,
                                                        active=self.view_types.index(self.view_type))
        self.radio_button_next_view_type.set_active(self.view_types.index(self.view_type))
        self.radio_button_next_camera_view.set_active(self.camera_views.index(self.camera_view))
        # self.slider_time = Slider(self.axis_slider_time, '', 0, options.n_bins - 1, valinit=self.time, valfmt='%d')

    def next(self, event=None, step=1):

        for i, event in zip(range(step), self.event_iterator):

            self.event_id = event.r0.event_id
            self.r0_container = event.r0.tel[self.telescope_id]
            self.r1_container = event.r1.tel[self.telescope_id]
            self.dl0_container = event.dl0.tel[self.telescope_id]
            self.dl1_container = event.dl1.tel[self.telescope_id]
            self.dl2_container = event.dl2


        self.data_r0 = np.array(list(self.r0_container.adc_samples.values()))

        if self.first_call:

            data = self.data_r0

            self.n_bins = data.shape[-1]
            self.readout_x = 4 * np.arange(0, self.n_bins, 1)
            self.trace_time_plot, = self.axis_readout.plot(np.array([self.time, self.time]) * 4, np.array([np.min(data[self.pixel_id]), np.max(data[self.pixel_id])]), color='r',
                                                          linestyle='--')
            self.trace_readout, = self.axis_readout.step(self.readout_x, data[self.pixel_id],
                                                         label='%s %d' % (self.view_type, self.pixel_id), where='mid')

        self.first_call = False
        self.local_time = self.r0_container.local_camera_clock
        self.update()

    def next_camera_view(self, camera_view, event=None):

        self.camera_view = camera_view
        if camera_view == 'p.e.':
            self.camera_visu.colorbar.set_label('[p.e.]')

        else:
            self.camera_visu.colorbar.set_label('[LSB]')
        self.update()

    def next_view_type(self, view_type, event=None):

        self.view_type = view_type
        self.update()

    def set_time(self, time):

        if time < self.n_bins and time >= 0:
            self.time = time
            self.update()

    def set_pixel(self, pixel_id):

        if pixel_id < 1296 and pixel_id >= 0:
            self.pixel_id = pixel_id
            self.update()

    def update(self):

        self.draw_readout(self.pixel_id)
        self.draw_camera()
        self.button_next_event.label.set_text('Next : current event #%d' % (self.event_id))
        # self.slider_time.set_val(self.time)

    def draw(self):

        self.next()
        self.button_next_event.on_clicked(self.next)
        self.radio_button_next_camera_view.on_clicked(self.next_camera_view)
        self.radio_button_next_view_type.on_clicked(self.next_view_type)
        # self.slider_time.on_changed(self.set_time)
        self.figure.canvas.mpl_connect('key_press_event', self.press)
        self.camera_visu._on_pick(self.event_clicked_on)
        plt.show()

    def draw_readout(self, pix):


        y = self.compute_trace()[pix]
        limits_y = self.limits_readout if self.limits_readout is not None else [np.min(y), np.max(y) + 10]
        # limits_y = [np.min(y), np.max(y) + 1]
        self.pixel_id = pix
        self.event_clicked_on.ind[-1] = self.pixel_id
        self.trace_readout.set_ydata(y)
        self.trace_readout.set_label('%s : %d, bin : %d' % (self.view_type, self.pixel_id, self.time))
        self.trace_time_plot.set_ydata(limits_y)
        self.trace_time_plot.set_xdata(self.time * 4)
        # y_ticks = np.linspace(np.min(y), np.max(y) + (np.max(y)-np.min(y)//10), np.max(y)-np.min(y)//10)
        # self.axis_readout.set_yticks(np.linspace(np.min(y), np.max(y), 8).astype(int))
        self.axis_readout.set_ylim(limits_y)
        self.axis_readout.legend(loc='upper right')


        # self.axis_readout.cla()
        # self.axis_readout.plot(np.array([self.time, self.time])*4, np.array([np.min(y), np.max(y)]), color='r', linestyle='--')
        # self.axis_readout.step(x, y,
        #                 label='%s %d' % (self.view_type, self.pixel_id), where='mid')

    def draw_camera(self):

        self.camera_visu.image = self.compute_image()

    def compute_trace(self):

        if self.view_type in self.view_types:

            if self.view_type == 'pixel':

                image = self.data_r0

            elif self.view_type == 'trigger_out':

                # to_print = ['%d, %d' % (pixel.ID, pixel.patch) for pixel in self.camera.Pixels]
                # print(to_print)

                image = np.array([self.trigger_output[pixel.patch] for pixel in self.camera.Pixels])

            elif self.view_type == 'trigger_in':

                image = np.zeros(self.data_r0.shape)
                print('%s not implemented' % self.view_type)

        return image

    def compute_image(self):

        image = self.compute_trace()
        # image = image - np.mean(image[..., -10:], axis=1)[:, np.newaxis]

        if self.camera_view in self.camera_views:

            if self.camera_view == 'mean':

                self.image = np.mean(image, axis=1)

            elif self.camera_view == 'std':

                self.image = np.std(image, axis=1)

            elif self.camera_view == 'max':

                self.image = np.max(image, axis=1)

            elif self.camera_view == 'sum':

                self.image = np.sum(image, axis=1)

            elif self.camera_view == 'time':

                self.image = image[:, self.time]

            elif self.camera_view == 'stacked':

                self.image += np.mean(image, axis=1).astype(int)

            elif self.camera_view == 'r1':

                self.image = self.r1_container.pe_samples

            elif self.camera_view == 'dl1' or self.camera_view == 'dl2':

                self.image = np.ma.masked_where(self.dl1_container.extracted_samples, self.dl1_container.image)

                if self.camera_view == 'dl2':

                    self.camera_visu.overlay_moments(self.dl2_container.shower)


        else:

            print('Cannot compute for camera type : %s' % self.camera_view)
        # print(np.max(self.image))
        return np.ma.masked_where(
            np.logical_or(self.image <= self.colorbar_limits[0], self.image >= self.colorbar_limits[1]), self.image)

    def press(self, event):

        sys.stdout.flush()

        if event.key == 'enter':
            self.next()

        if event.key == 'right':
            self.set_time(self.time + 1)

        if event.key == 'left':
            self.set_time(self.time - 1)

        if event.key == '+':
            self.set_pixel(self.pixel_id + 1)

        if event.key == '-':
            self.set_pixel(self.pixel_id - 1)

        if event.key == 'h':
            self.axis_next_event_button.set_visible(False)

        if event.key == 'v':
            self.axis_next_event_button.set_visible(True)

        self.update()

    def save(self, filename='test.png'):

        self.set_buttons_visible(False)
        self.figure.savefig(filename)
        self.set_buttons_visible(True)

        return self.figure

    def set_buttons_visible(self, visible=True):

        self.axis_next_camera_view_button.set_visible(visible)
        self.axis_next_view_type_button.set_visible(visible)
        # self.axis_slider_time.set_visible(visible)
        self.axis_next_event_button.set_visible(visible)

    def animate_pixel_scan(self, pixel_list, filename='test.mp4'):

        self.set_buttons_visible(visible=False)

        metadata = dict(title='Mapping Scan', artist='Digicam Film Studio')
        writer = animation.FFMpegWriter(fps=20, metadata=metadata)

        # next_event = lambda i: self.next(event=None, index=i)

        with writer.saving(self.figure, filename, 100):

            # print(pixel_list)
            # print(len(pixel_list))

            for i, pixel in enumerate(pixel_list[:-1]):

                try:

                    self.pixel_id = pixel_list[i]
                    # self.update()
                    self.next()
                    writer.grab_frame()

                except:

                    break

        self.set_buttons_visible(visible=True)


        # ani = animation.FuncAnimation(self.figure, next_event, np.arange(0, 10, 1), blit=True, interval=10,
        #                        repeat=False)

        # ani.save(filename, metada={'studio': 'DigiCam Films Production'})

    def animate_muon_scan(self, filename='muon.mp4', n_frames=10):

        self.set_buttons_visible(visible=False)

        metadata = dict(title='High threshold events', artist='Digicam Film Studio')
        writer = animation.FFMpegWriter(fps=10, metadata=metadata)

        # next_event = lambda i: self.next(event=None, index=i)

        with writer.saving(self.figure, filename, 100):

            # print(pixel_list)
            # print(len(pixel_list))

            for i in enumerate(range(n_frames)):

                try:

                    self.next()
                    index_max = np.argmax(self.data)
                    index_max = np.unravel_index(index_max, self.data.shape)
                    self.pixel_id = index_max[0]
                    self.time = index_max[1]
                    self.update()
                    writer.grab_frame()


                except:

                    break

        self.set_buttons_visible(visible=True)


        # ani = animation.FuncAnimation(self.figure, next_event, np.arange(0, 10, 1), blit=True, interval=10,
        #                        repeat=False)

        # ani.save(filename, metada={'studio': 'DigiCam Films Production'})

    def heat_map_animation(self, filename='hit_map.mp4', n_frames=500, limits_colormap=None):

        # self.camera_view = 'std'

        self.set_buttons_visible(visible=False)

        self.figure = plt.figure(figsize=(10, 10))

        self.axis_camera = self.figure.add_subplot(111)
        self.axis_camera.axis('off')
        self.camera_visu = visualization.CameraDisplay(self.geometry, ax=self.axis_camera, title='', norm=self.scale,
                                                       cmap='viridis',
                                                       allow_pick=True)
        if limits_colormap is not None:
            self.camera_visu.set_limits_minmax(limits_colormap[0], limits_colormap[1])

        self.camera.image = self.compute_image()
        # self.camera_visu.cmap.set_bad(color='w')
        self.camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05, shrink=.85)

        if self.scale == 'log':
            self.camera_visu.colorbar.set_norm(LogNorm(vmin=1, vmax=None, clip=False))
        self.camera_visu.colorbar.set_label('[ADC]')
        self.camera_visu.axes.get_xaxis().set_visible(False)
        self.camera_visu.axes.get_yaxis().set_visible(False)
        self.camera_visu.on_pixel_clicked = self.draw_readout

        metadata = dict(title='High threshold events', artist='Digicam Film Studio')
        writer = animation.FFMpegWriter(fps=10, metadata=metadata)

        # next_event = lambda i: self.next(event=None, index=i)

        with writer.saving(self.figure, filename, 100):

            # print(pixel_list)
            # print(len(pixel_list))

            for i in enumerate(range(n_frames)):
                # try:

                # print('hello')
                self.next()

                # print(np.max(self.image))

                # if np.max(self.image)<=self.threshold:

                self.draw_camera()
                writer.grab_frame()


                # except:

                # break

        self.set_buttons_visible(visible=True)


class Event_Clicked():
    def __init__(self, pixel_start):
        self.ind = [0, pixel_start]
