import numpy as np
import pandas as pd

def get_addon_feat_polars(df, stage, train=False):
    
    # Time per level and event
    COLS = []
    EVENT_AT_LEVEL_STG1 = {
        'level0_events' : ['navigate_click', 'notification_click', 'object_click', 'person_click'],
        'level1_events' : ['cutscene_click', 'navigate_click', 'object_click'],
        'level2_events' : ['cutscene_click', 'navigate_click', 'notification_click', 'object_click'],
        'level3_events' : ['cutscene_click', 'navigate_click', 'notification_click', 'object_click', 'map_click'],
        'level4_events' : ['navigate_click', 'map_click']
    }

    EVENT_AT_LEVEL_STG2 = {
        'level5_events' : ['cutscene_click', 'map_click', 'navigate_click'],
        'level6_events' : ['cutscene_click', 'navigate_click', 'observation_click', 'person_click'],
        'level7_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click', 'person_click'],
        'level8_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click', 'person_click'],
        'level9_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click', 'person_click'],
        'level10_events' : ['navigate_click', 'notification_click', 'object_click', 'person_click'],
        'level11_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click'],
        'level12_events' : ['map_click', 'navigate_click']
    }

    EVENT_AT_LEVEL_STG3 = {
        'level13_events' : ['cutscene_click', 'map_click', 'navigate_click'],
        'level14_events' : ['navigate_click'],
        'level15_events' : ['person_click'],
        'level16_events' : ['cutscene_click', 'navigate_click'],
        'level17_events' : ['cutscene_click', 'navigate_click'],
        'level18_events' : ['person_click', 'map_click', 'navigate_click', 'notification_click', 'object_click'],
        'level19_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click'],
        'level20_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click'],
        'level21_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click'],
        'level22_events' : ['map_click', 'navigate_click']
    }


    tmp = df.groupby(['session_id', 'level', 'event_name']).agg({'action_time' : ['sum', 'mean', 'std', 'median', 'max']}).reset_index()
    tmp.columns = tmp.columns.map(''.join)
    _train = tmp.pivot_table(index='session_id', columns=['level', 'event_name'], values=['action_timesum', 'action_timemean', 'action_timestd', 'action_timemedian', 'action_timemax'])
    _train.columns = [str(i[1]) + '_' + i[2] + '_' + i[0] for i in _train.columns]

    if stage == 1:
        level_range = range(0,5)
        EVENT_AT_LEVEL = EVENT_AT_LEVEL_STG1
    elif stage == 2:
        level_range = range(5,13)
        EVENT_AT_LEVEL = EVENT_AT_LEVEL_STG2
    else:
        level_range = range(13,23)
        EVENT_AT_LEVEL = EVENT_AT_LEVEL_STG3

    for level, cols in zip(list(level_range), EVENT_AT_LEVEL.values()):
        COLS.extend([str(level) + '_' + i + '_' + 'action_timesum' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemean' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timestd' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemedian' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemax' for i in cols])

    cols_drop_no_std = ['0_notification_click_action_timestd', '3_map_click_action_timestd', '4_map_click_action_timestd', '5_map_click_action_timestd', '6_observation_click_action_timestd', \
                        '7_map_click_action_timestd', '7_notification_click_action_timestd', '8_map_click_action_timestd', '9_map_click_action_timestd', '10_notification_click_action_timestd',\
                        '10_object_click_action_timestd', '11_map_click_action_timestd', '12_map_click_action_timestd', '13_map_click_action_timestd', '18_map_click_action_timestd', \
                        '18_notification_click_action_timestd', '19_map_click_action_timestd', '19_notification_click_action_timestd', '20_notification_click_action_timestd', \
                        '20_map_click_action_timestd', '21_map_click_action_timestd', '22_map_click_action_timestd']

    COLS = np.array(COLS)[~np.isin(COLS, cols_drop_no_std)]

    if train == False:
        missing_cols = np.array(COLS)[~np.isin(COLS, _train.columns)]
        for _col in missing_cols:
            _train[_col] = 0

    _train = _train[COLS].reset_index()

    
    # Time per level and room
    COLS = []
    ROOM_AT_LEVEL_STG1 = {
            'level0_rooms' : ['tunic.historicalsociety.closet'],
            'level1_rooms' : ['tunic.historicalsociety.basement', 'tunic.historicalsociety.closet', 'tunic.historicalsociety.entry'],
            'level2_rooms' : ['tunic.historicalsociety.collection', 'tunic.historicalsociety.entry'],
            'level3_rooms' : ['tunic.historicalsociety.collection', 'tunic.historicalsociety.entry', 'tunic.kohlcenter.halloffame'],
            'level4_rooms' : ['tunic.capitol_0.hall', 'tunic.kohlcenter.halloffame']
        }

    ROOM_AT_LEVEL_STG2 = {
            'level5_rooms' : ['tunic.capitol_0.hall', 'tunic.historicalsociety.basement', 'tunic.historicalsociety.closet_dirty', 'tunic.historicalsociety.entry'],
            'level6_rooms' : ['tunic.historicalsociety.basement', 'tunic.historicalsociety.closet_dirty','tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk',\
                              'tunic.historicalsociety.stacks'],
            'level7_rooms' : ['tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk', 'tunic.historicalsociety.stacks', 'tunic.humanecology.frontdesk'],
            'level8_rooms' : ['tunic.drycleaner.frontdesk', 'tunic.humanecology.frontdesk'],
            'level9_rooms' : ['tunic.drycleaner.frontdesk', 'tunic.library.frontdesk', 'tunic.library.microfiche'],
            'level10_rooms' : ['tunic.library.frontdesk', 'tunic.library.microfiche'],
            'level11_rooms' : ['tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk', 'tunic.historicalsociety.stacks', 'tunic.library.frontdesk'],
            'level12_rooms' : ['tunic.capitol_1.hall', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.stacks']
        }

    ROOM_AT_LEVEL_STG3 = {
            'level13_rooms' : ['tunic.capitol_1.hall', 'tunic.historicalsociety.basement', 'tunic.historicalsociety.entry'],
            'level14_rooms' : ['tunic.historicalsociety.basement', 'tunic.historicalsociety.cage'],
            'level15_rooms' : ['tunic.historicalsociety.basement', 'tunic.historicalsociety.cage', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk','tunic.historicalsociety.stacks'],
            'level16_rooms' : ['tunic.historicalsociety.basement', 'tunic.historicalsociety.cage', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk', 'tunic.historicalsociety.stacks'],
            'level17_rooms' : ['tunic.historicalsociety.basement', 'tunic.historicalsociety.cage', 'tunic.historicalsociety.collection_flag', 'tunic.historicalsociety.entry'],
            'level18_rooms' : ['tunic.historicalsociety.collection_flag', 'tunic.historicalsociety.entry', 'tunic.wildlife.center'],
            'level19_rooms' : ['tunic.flaghouse.entry', 'tunic.wildlife.center'],
            'level20_rooms' : ['tunic.flaghouse.entry', 'tunic.library.frontdesk', 'tunic.library.microfiche'],
            'level21_rooms' : ['tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk', 'tunic.historicalsociety.stacks', 'tunic.library.frontdesk', 'tunic.library.microfiche'],
            'level22_rooms' : ['tunic.capitol_2.hall', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.stacks']
        }



    tmp = df.groupby(['session_id', 'level', 'room_fqid']).agg({'action_time' : ['sum', 'mean', 'std', 'median', 'max']})
    tmp.columns = tmp.columns.map(''.join)
    tmp_pivot = tmp.pivot_table(index='session_id', columns=['level', 'room_fqid'], values=['action_timesum', 'action_timemean', 'action_timestd', 'action_timemedian', 'action_timemax'])
    tmp_pivot.columns = [str(i[1]) + '_' + i[2] + '_' + i[0] for i in tmp_pivot.columns]
    
    if stage == 1:
        level_range = range(0,5)
        ROOM_AT_LEVEL = ROOM_AT_LEVEL_STG1
    elif stage == 2:
        level_range = range(5,13)
        ROOM_AT_LEVEL = ROOM_AT_LEVEL_STG2
    else:
        level_range = range(13,23)
        ROOM_AT_LEVEL = ROOM_AT_LEVEL_STG3
    
    for level, cols in zip(list(level_range), ROOM_AT_LEVEL.values()):
        COLS.extend([str(level) + '_' + i + '_' + 'action_timesum' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemean' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timestd' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemedian' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemax' for i in cols])

    cols_drop_no_std = ['5_tunic.historicalsociety.basement_action_timestd', '5_tunic.historicalsociety.entry_action_timestd', '6_tunic.historicalsociety.basement_action_timestd',\
                        '6_tunic.historicalsociety.entry_action_timestd', '6_tunic.historicalsociety.stacks_action_timestd', '7_tunic.historicalsociety.stacks_action_timestd',\
                        '11_tunic.historicalsociety.entry_action_timestd', '13_tunic.historicalsociety.entry_action_timestd', '14_tunic.historicalsociety.basement_action_timestd',\
                        '15_tunic.historicalsociety.basement_action_timestd', '15_tunic.historicalsociety.stacks_action_timestd', '16_tunic.historicalsociety.basement_action_timestd',\
                        '16_tunic.historicalsociety.entry_action_timestd', '16_tunic.historicalsociety.stacks_action_timestd', '21_tunic.historicalsociety.entry_action_timestd']    
    COLS = np.array(COLS)[~np.isin(COLS, cols_drop_no_std)]
    
    if train == False:
        missing_cols = np.array(COLS)[~np.isin(COLS, tmp_pivot.columns)]
        for _col in missing_cols:
            tmp_pivot[_col] = 0

    tmp_pivot = tmp_pivot[COLS].reset_index()

    
    _train = pd.merge(left=_train, right=tmp_pivot, on='session_id', how='left')
    
    return _train