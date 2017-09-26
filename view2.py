from digicamviewer import viewer
from digicampipe import event_stream

if __name__ == '__main__':

    directory = '/home/alispach/Downloads/'

    camera_config_file = '/home/alispach/Documents/PhD/ctasoft/CTS/config/camera_config.cfg'
    file_basename = directory + 'CameraDigicam@sst1mserver_0_000.%d.fits.fz'
    file_list = [file_basename % i for i in range(130, 136)]
    print(file_list)
    data_stream = event_stream.event_stream(file_list=file_list, expert_mode=True)

    display = viewer.EventViewer2(data_stream, n_samples=50, camera_config_file=camera_config_file)#, scale='lin')
    display.next(step=1)
    display.draw()
