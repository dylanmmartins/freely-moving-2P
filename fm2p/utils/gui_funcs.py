# -*- coding: utf-8 -*-
"""
GUI helper functions.

Example usages
--------------

    cfg_path = fm2p.select_file(
        title='Select config yaml file.',
        filetypes=[('YAML','.yaml'),('YML','.yml'),]
    )

    rec_dir = fm2p.select_directory(
        title='Select the directory containing the recordings.'
    )

    user_input = fm2p.get_string_input(
        title='Enter a string input.'
    )

Functions
---------
select_file(title, filetypes)
    Select a file using a file dialog.
select_directory(title)
    Select a directory using a file dialog.
get_string_input(title)
    Get a string input from the user using a dialog.

Author: DMM, 2024
"""


import tkinter as tk
from tkinter import filedialog


def select_file(title, filetypes):
    """ Select a file using a file dialog.
    
    Parameters
    ----------
    title : str
        Title of the file dialog.
    filetypes : list of tuples
        List of file types to filter the files shown in the dialog.
        e.g., [('Text files', '*.txt'), ('All files', '*.*')]
    
    Returns
    -------
    file_path : str
        The path to the selected file.
    """

    print(title)
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    print(file_path)

    return file_path


def select_directory(title):
    """ Select a directory using a file dialog.
    
    Parameters
    ----------
    title : str
        Title of the directory dialog.
    
    Returns
    -------
    directory_path : str
        The path to the selected directory.
    """

    print(title)
    root = tk.Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory(
        title=title,
    )
    print(directory_path)

    return directory_path


def get_string_input(title):
    """ Get a string input from the user using a dialog.
    
    Parameters
    ----------
    title : str
        Title of the input dialog.
    
    Returns
    -------
    user_input : str
        The string input provided by the user.
    """

    print(title)

    root = tk.Tk()
    label = tk.Label(root, text=title)
    root.minsize(width=300, height=20)
    root.title(title)
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    user_input = None

    def retrieve_input():
        nonlocal user_input
        user_input = entry.get()
        root.destroy()
        
    button = tk.Button(root, text='Enter', command=retrieve_input)
    button.pack()

    root.bind("<Return>", lambda event: retrieve_input())

    root.mainloop()

    print(user_input)

    return user_input

