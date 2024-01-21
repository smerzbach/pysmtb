import difflib
from pathlib import Path
from typing import *


def list_all_experiments(
        experiment_folders: List[Union[str, Path]],
        allowed_suffix: Optional[str] = None,
) -> list:
    """
    List all experiments in a list of experiment folders.

    :param experiment_folders: List of experiment folders
    :param allowed_suffix: Optional suffix to filter experiments events files by, e.g. .0 for events.*.0
    """

    experiments = []
    for d in experiment_folders:
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file() and f.name.startswith('events.'):
                    if allowed_suffix is None or f.name.endswith(allowed_suffix):
                        experiments.append(f)
    experiments = sorted(list(set(experiments)))
    return experiments


def visualize_str_diff_html(
        str_ref: str,
        str_comp: str,
        prefix_ref: str = '',
        prefix_comp: str = '',
        color_insert: str = '#00ff00',
        color_delete: str = '#ff0000',
        color_replace: str = '#ffff00',
) -> str:
    """
    Create HTML showing the diff of two strings via colored text.
    :param str_ref: Reference string
    :param str_comp: String to compare to reference
    :param prefix_ref: Prefix for reference string
    :param prefix_comp: Prefix for comparison string
    :param color_insert: Color for inserted text
    :param color_delete: Color for deleted text
    :param color_replace: Color for replaced text
    :return: HTML string with colorized differences
    """
    codes = difflib.SequenceMatcher(a=str_ref, b=str_comp, autojunk=False).get_opcodes()

    diff_str = '<div style="background-color: #cccccc;">' + prefix_ref + ' ' + str_ref + '<br>\n' + prefix_comp + ' '
    for code in codes:
        if code[0] == 'equal':
            diff_str += str_ref[code[1]:code[2]]
        elif code[0] == 'insert':
            diff_str += f'<span style="background-color: {color_insert}">' + str_comp[code[3]:code[4]] + '</span>'
        elif code[0] == 'delete':
            diff_str += f'<span style="background-color: {color_delete}">' + str_ref[code[1]:code[2]] + '</span>'
        elif code[0] == 'replace':
            diff_str += f'<span style="background-color: {color_replace}">' + str_comp[code[3]:code[4]] + '</span>'
        else:
            raise NotImplementedError(f'unknown code {code[0]}')
    diff_str += '</div>\n'
    return diff_str
