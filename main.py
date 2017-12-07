from bokeh.io import show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LogColorMapper
)
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
import numpy as np
from cts_core.camera import Camera
import cts_core.geometry as geometry
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.layouts import widgetbox
from bokeh.layouts import row
from bokeh.models import LogColorMapper, LogTicker, ColorBar
import pandas as pd
from bokeh.models.widgets import PreText, Select

# palette.reverse()
color_mapper = LogColorMapper(palette=palette)

camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
digicam = Camera(_config_file=camera_config_file)
directory = '/home/alispach/data/CRAB_01/'  #
nsb_filename = 'nsb.npz'
dark_baseline = np.load(directory + 'dark.npz')
nsb_baseline = np.load(directory + nsb_filename)

data = pd.DataFrame()
data['x'] = [geometry.createPixel(pixel.center[0], pixel.center[1])[0] for pixel in digicam.Pixels]
data['x_center'] = [pixel.center[0] for pixel in digicam.Pixels]
data['y'] = [geometry.createPixel(pixel.center[0], pixel.center[1])[1] for pixel in digicam.Pixels]
data['y_center'] = [pixel.center[1] for pixel in digicam.Pixels]
data['pixel_id'] = [pixel.ID for pixel in digicam.Pixels]
data['text'] = [str(pixel.ID) for pixel in digicam.Pixels]
data['in_patch_id'] = [pixel.id_inPatch for pixel in digicam.Pixels]
data['in_module_id'] = [pixel.id_inModule for pixel in digicam.Pixels]
data['in_fadc_id'] = [pixel.id_inFADC for pixel in digicam.Pixels]
data['in_crate_id'] = [pixel.in_inCrate for pixel in digicam.Pixels]
data['fadc_quad_id'] = [pixel.fadcQuad for pixel in digicam.Pixels]
data['fadc_channel_id'] = [pixel.fadcChannel for pixel in digicam.Pixels]
data['sector_id'] = [pixel.sector for pixel in digicam.Pixels]
data['patch_id'] = [pixel.patch for pixel in digicam.Pixels]
data['n_pixels'] = len(data['pixel_id'])
data['dark_baseline'] = dark_baseline['baseline']


n_bins = nsb_baseline['baseline'].shape[0]

test = {'baseline_{}'.format(pixel_id): value for pixel_id, value in enumerate(nsb_baseline['baseline'].T)}
data_1 = pd.DataFrame(data=test, index=np.arange(n_bins))
data_1['time'] = nsb_baseline['time_stamp']
print(data_1)


# df = pd.DataFrame(data=data)

source = ColumnDataSource(data=data)
source_meta = ColumnDataSource(data=data_1)

tools = ['pan', 'wheel_zoom', 'reset', 'hover', 'save', 'tap', 'poly_select']

p = figure(plot_width=600, plot_height=600, title="Hardware viewer", tools=tools, toolbar_location='above')
p.patches(xs='x', ys='y', source=source, fill_color={'field': 'dark_baseline', 'transform': color_mapper})

color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                     label_standoff=12, border_line_color=None, location=(0, 0))

p.add_layout(color_bar, 'right')
#p.text(x='x_center', y='y_center', text='text', source=source, angle=0, text_font_size=[0.1]*n_pixels,
#       x_offset=[-5]*n_pixels, y_offset=[5]*n_pixels)
hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [("pixel", "@pixel_id"), ("(x, y)", "(@x_center, @y_center) mm"), ('sector', '@sector_id'),
                  ('patch', '@patch_id'), ('in fadc', '@in_fadc_id'), ('in crate', '@in_crate_id'),
                  ('fadc quad', '@fadc_quad_id'), ('fadc channel', '@fadc_channel_id')]


columns = [TableColumn(field=key, title=key) for key in source.data.keys()]
data_table = DataTable(source=source, columns=columns, width=600, height=600)



p_1 = figure(plot_width=400, plot_height=400)

# add a line renderer
p_1.line('time', 'baseline_1', source=data_1, line_color='red')



stats = PreText(text='', width=500)
ticker1 = Select(value='baseline', options=['baseline', 'std'])

# show(row(widgetbox(data_table), p, p_1, ticker1))
show(row(p, p_1, ticker1))
# show()
# show(p)
