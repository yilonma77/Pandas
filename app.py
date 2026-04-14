import tkinter as tk
from tkinter import messagebox

def on_ok():
    messagebox.showinfo("Message", "Bienvenue !")

root = tk.Tk()
root.title("Bienvenue App")
root.geometry("260x120")
root.resizable(False, False)

# Centrer la fenêtre
root.eval("tk::PlaceWindow . center")

label = tk.Label(root, text="Cliquez sur OK pour continuer", pady=16)
label.pack()

btn = tk.Button(root, text="OK", width=12, height=2, command=on_ok)
btn.pack()

root.mainloop()
