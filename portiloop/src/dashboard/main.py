import random
import time
from nicegui import ui
import numpy as np
from matplotlib import pyplot as plt
from portiloop.src.utils import ADSFrontend, get_portiloop_version
from portiloop.src.capture import capture_process
from portiloop.src import ADS
if ADS:
    from portiloop.src.hardware.frontend import Frontend

timediff = 1/250
index = 0


def get_frontend():
    version = get_portiloop_version()

    # Check which version of the ADS we are in.
    nb_channels = 0
    if version != -1:
        nb_channels = Frontend(version).get_version()

    channel_states = [
        'simple'
    ] * nb_channels

    capture_frontend = ADSFrontend(
        duration=28800, 
        frequency=250,
        python_clock=True,
        channel_states=channel_states, 
        process=capture_process
    )

    return capture_frontend


columns = [
    {'name': 'name', 'label': 'Name', 'field': 'name', 'required': True},
    {'name': 'duration', 'label': 'Duration (s)',
        'field': 'duration', 'sortable': True},
]
rows = [
    {'id': 0, 'name': 'Example Label', 'duration': 5.0},
]

ads_frontend = None
def start_frontend():
    global ads_frontend
    dialog.close()
    ads_frontend = get_frontend()
    ads_frontend.init_capture()
    ui.timer(timediff, get_next_window)

def get_next_window():
    global index
    global ads_frontend
    point = ads_frontend.get_data()
    index += 1
    if point is not None:
        line_plot.push([index], [[point[0][0]]])

def start_rec_callback(e):
    # Start recording here
    pass
    # container.remove(2)
    # label = ui.icon('radio_button_checked',
    #                 color='primary').classes('text-5xl')


def stop_rec_callback(e):
    # Stop recording here
    pass
    # container.remove(2)
    # label = ui.icon('radio_button_unchecked',
    #                 color='primary').classes('text-5xl')


with ui.dialog() as dialog, ui.card():
    ui.label('Start communication with the ADS:')
    ui.button('Start', on_click=start_frontend)

dialog.open()

with ui.tabs().classes('w-full') as tabs:
    labelling_tab = ui.tab('Labelling')
    labels = ui.tab('Labels')

with ui.tab_panels(tabs, value=labelling_tab).classes('w-full'):
    with ui.tab_panel(labelling_tab):
        with ui.row():
            with ui.card().classes('w-2/3'):
                line_plot = ui.line_plot(
                    n=1, limit=250, figsize=(10, 10), update_every=10).classes('w-full')
                container = ui.row()
                with container.classes('w-full justify-evenly'):
                    with ui.row().classes('w-1/2 justify-start items-start'):
                        ui.button('Start Recording', on_click=start_rec_callback,
                                  icon='play_circle')
                        ui.button('Stop Recording', on_click=stop_rec_callback,
                                  icon='stop').props('flat')
                    with ui.row():
                        ui.button('Start Label', icon='flag').classes(
                            'justify-end items-end')
                        ui.button('Stop Label').props(
                            'flat').classes('justify-end items-end')

            with ui.card().classes('w-auto'):
                with ui.table(title='Labels', columns=columns, rows=rows, selection='single', pagination=10).classes('w-96') as table:
                    with table.add_slot('top-right'):
                        with ui.input(placeholder='Search').props('type=search').bind_value(table, 'filter').add_slot('append'):
                            ui.icon('search')
                    with table.add_slot('bottom-row'):
                        with table.row():
                            with table.cell():
                                ui.button(on_click=lambda: (
                                    table.add_rows(
                                        {'id': time.time(), 'name': new_name.value, 'duration': new_duration.value}),
                                    new_name.set_value(None),
                                    new_duration.set_value(None),
                                ), icon='add').props('flat fab-mini')
                            with table.cell():
                                new_name = ui.input('Name')
                            with table.cell():
                                new_duration = ui.number('duration')
                ui.button('Remove', on_click=lambda: table.remove_rows(*table.selected), icon='delete') \
                    .bind_visibility_from(table, 'selected', backward=lambda val: bool(val))

    with ui.tab_panel(labels):
        ui.label('Second tab')

ui.run(title='Portiloop Dashboard', favicon='🧠')