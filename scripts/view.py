from digicamviewer import viewer
from digicampipe import event_stream

if __name__ == '__main__':

    directory = '/home/alispach/blackmonkey/'

    camera_config_file = '/home/alispach/Documents/PhD/ctasoft/CTS/config/camera_config.cfg'
    file_basename = directory + 'CameraDigicam@sst1mserver_0_000.%d.fits.fz'
    file_list = [file_basename % i for i in range(50, 100)]
    data_stream = event_stream.event_stream(file_list=file_list, expert_mode=True)

    display = viewer.EventViewer(data_stream, camera_config_file=camera_config_file)
    display.draw()