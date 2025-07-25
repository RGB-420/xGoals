import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import os
import time

# Columns for data (removed key_pass_length and key_pass_angle)
columns = [
    'match', 'period', 'minute', 'second', 'shot_location', 'pass_location', 'gk_location',
    'under_pressure', 'play_pattern_id', 'shot_key_pass_id', 'shot_type_id',
    'shot_technique_id', 'shot_outcome_id', 'shot_body_part_id', 'players_in_range',
    'shot_first_time', 'shot_aerial_won', 'key_pass_height_id', 'key_pass_body_part_id',
    'key_under_pressure', 'key_pass_switch', 'key_pass_cross', 'key_pass_cut_back',
    'possession_duration'
]

# Placeholder dictionaries for reference (user to fill)
play_pattern_dict = {1:	'Regular Play',2:'From Corner',3: 'From Free Kick',4: 'From Throw In', 5: 'Other', 6:	'From Counter', 7: 'From Goal Kick',8: 'From Keeper',9: 'From Kick Off'}
shot_type_dict = {61: 'Corner', 62: 'Free Kick', 65: 'Kick Off', 87: 'Open Play', 88: 'Penalty'}
shot_technique_dict = {89: 'Backheel', 90: 'Diving Header', 91: 'Half Volley', 92: 'Lob', 93: 'Normal', 94:	'Overhead Kick', 95: 'Volley'}
shot_outcome_dict = {96: 'Blocked', 97:	'Goal', 98:	'Off T', 99: 'Post', 100: 'Saved', 101:	'Wayward', 115:	'Saved Off Target', 116: 'Saved to Post'}
shot_body_part_dict = {37: 'Head', 38: 'Left Foot', 40: 'Right Foot', 70: 'Other'}
key_pass_height_dict = {1: 'Ground Pass', 2: 'Low Pass', 3: 'High Pass'}
key_pass_body_part_dict = {37: 'Head',38: 'Left Foot', 40: 'Right Foot', 68: 'Drop Kick', 69: 'Keeper Arm', 70:	'Other', 106: 'No Touch'}

# Parse coordinate string
def parse_xy(text):
    try:
        x, y = text.split(',')
        return float(x.strip()), float(y.strip())
    except:
        return None

# Load or initialize DataFrame
data_file = 'output/events.csv'
if os.path.exists(data_file):
    df = pd.read_csv(data_file)
else:
    df = pd.DataFrame(columns=columns)

class MatchRecorder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.bell = lambda *args, **kwargs: None
        # Style
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Treeview', rowheight=24, font=('Arial', 10))
        style.configure('Treeview.Heading', font=('Arial', 11, 'bold'))
        style.configure('TLabel', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10))
        style.configure('TEntry', font=('Arial', 10))
        style.configure('TCombobox', font=('Arial', 10))

        self.title('Football Match Event Recorder')
        self.geometry('1400x900')
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        # DataFrame
        if os.path.exists(data_file):
            self.df = pd.read_csv(data_file)
        else:
            self.df = pd.DataFrame(columns=columns)

        # Variables
        self.widgets = {}
        self.timer_running = False
        self.start_time = 0
        self.elapsed_time = 0

        # Load assets
        self.load_image()

        # Build UI
        self.create_layout()
        self.load_table()

    def load_image(self):
        path = 'utils/soccer_field.png'
        if not os.path.exists(path):
            messagebox.showerror('Error', f'Image missing: {path}', parent=self)
            self.destroy(); return
        img = Image.open(path)
        max_w, max_h = 350, 250
        w, h = img.size
        r = min(max_w/w, max_h/h)
        img = img.resize((int(w*r), int(h*r)), Image.LANCZOS)
        self.pitch_img = ImageTk.PhotoImage(img)

    def create_layout(self):
        # Main frames
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        mid_frame = ttk.Frame(self)
        mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Match Info and Timer/Actions
        match_panel = ttk.LabelFrame(top_frame, text='Match Info')
        match_panel.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)
        ttk.Label(match_panel, text='Match').grid(row=0, column=0, sticky='w')
        e = ttk.Entry(match_panel, width=20)
        e.grid(row=0, column=1)
        self.widgets['match'] = e

        timer_panel = ttk.LabelFrame(top_frame, text='Timer & Actions')
        timer_panel.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=5)
        self.timer_label = ttk.Label(timer_panel, text='Duration: 0.00s')
        self.timer_label.grid(row=0, column=0, padx=5)
        ttk.Button(timer_panel, text='Start', command=self.start_timer).grid(row=0, column=1)
        ttk.Button(timer_panel, text='Stop', command=self.stop_timer).grid(row=0, column=2)
        ttk.Button(timer_panel, text='Reset', command=self.reset_timer).grid(row=0, column=3)
        ttk.Button(timer_panel, text='Add Event', command=self.add_record).grid(row=0, column=4, padx=10)
        ttk.Button(timer_panel, text='Save CSV', command=self.save_csv).grid(row=0, column=5)

        # Middle: split into three columns
        # Left column: Event Data
        left_frame = ttk.Frame(mid_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        input_panel = ttk.LabelFrame(left_frame, text='Event Data')
        input_panel.pack(fill=tk.Y, expand=False)
        self.build_inputs(input_panel)

        # Middle column: Pitch
        center_frame = ttk.Frame(mid_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        pitch_panel = ttk.LabelFrame(center_frame, text='Pitch (Click)')
        pitch_panel.pack(fill=tk.BOTH, expand=False)
        canv = tk.Canvas(pitch_panel, width=self.pitch_img.width(), height=self.pitch_img.height(), highlightthickness=0)
        canv.pack()
        canv.create_image(0,0,anchor=tk.NW,image=self.pitch_img)
        canv.bind('<Button-1>', lambda e: self.on_click(e,'shot_location'))
        canv.bind('<Button-3>', lambda e: self.on_click(e,'pass_location'))
        canv.bind('<Button-2>', lambda e: self.on_click(e,'gk_location'))

        # Right column: Reference Tables
        right_frame = ttk.Frame(mid_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ref_panel = ttk.LabelFrame(right_frame, text='Reference Tables')
        ref_panel.pack(fill=tk.BOTH, expand=True)
        self.create_reference_tables(ref_panel)

        # Bottom: Events Table
        table_panel = ttk.LabelFrame(bottom_frame, text='Recorded Events')
        table_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree = ttk.Treeview(table_panel, columns=columns, show='headings')
        for c in columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=80)
        vsb = ttk.Scrollbar(table_panel, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        canv.pack()
        canv.create_image(0,0,anchor=tk.NW,image=self.pitch_img)
        canv.bind('<Button-1>', lambda e:self.on_click(e,'shot_location'))
        canv.bind('<Button-3>', lambda e:self.on_click(e,'pass_location'))
        canv.bind('<Button-2>', lambda e:self.on_click(e,'gk_location'))

        # Bottom: Events Table
        table_panel = ttk.LabelFrame(bottom_frame, text='Recorded Events')
        table_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree = ttk.Treeview(table_panel, columns=columns, show='headings')
        for c in columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=80)
        vsb = ttk.Scrollbar(table_panel, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def build_inputs(self, parent):
        row=0
        # Period
        ttk.Label(parent,text='Period').grid(row=row, column=0, sticky='w');
        cmb = ttk.Combobox(parent,values=[1, 2, 3, 4, 5],width=18)
        cmb.grid(row=row, column=1); self.widgets['period']=cmb; row+=1
        fields=[
            ('Minute','minute'),('Second','second'),
            ('Shot Loc','shot_location'),('Pass Loc','pass_location'),('GK Loc','gk_location'),
            ('Play Pattern','play_pattern_id'),
            ('Shot Type','shot_type_id'),('Shot Technique','shot_technique_id'),
            ('Shot Outcome','shot_outcome_id'),('Shot Body Part','shot_body_part_id'),
            ('Players','players_in_range'),('Pass Height','key_pass_height_id'),
            ('Pass Body Part','key_pass_body_part_id'),('Possession(s)','possession_duration')
        ]
        for lbl,key in fields:
            ttk.Label(parent,text=lbl).grid(row=row,column=0,sticky='w');
            ent=ttk.Entry(parent,width=20); ent.grid(row=row,column=1);
            self.widgets[key]=ent; row+=1
        bools=[('Under Pressure','under_pressure'),('First Time','shot_first_time'),('Shot Key Pass','shot_key_pass_id'),
               ('Aerial Won','shot_aerial_won'),('Pass Under Pressure','key_under_pressure'),
               ('Pass Switch','key_pass_switch'),('Pass Cross','key_pass_cross'),('Pass Cut Back','key_pass_cut_back')]
        for lbl,key in bools:
            var=tk.BooleanVar(); chk=ttk.Checkbutton(parent,text=lbl,variable=var)
            chk.grid(row=row,column=0,columnspan=2,sticky='w'); self.widgets[key]=var; row+=1

    def create_reference_tables(self,parent):
        dicts=[
            ('Play Pattern',play_pattern_dict),
            ('Shot Type',shot_type_dict),('Shot Technique',shot_technique_dict),
            ('Shot Outcome',shot_outcome_dict),('Shot Body Part',shot_body_part_dict),
            ('Key Pass Height',key_pass_height_dict),('Key Pass Body Part',key_pass_body_part_dict)
        ]
        for idx,(title,d) in enumerate(dicts):
            fr=ttk.LabelFrame(parent,text=title); fr.grid(row=idx//2,column=idx%2,sticky='nsew',padx=3,pady=3)
            parent.grid_columnconfigure(idx%2,weight=1); parent.grid_rowconfigure(idx//2,weight=1)
            tv=ttk.Treeview(fr,columns=['ID','Desc'],show='headings',height=4)
            tv.heading('ID',text='ID'); tv.heading('Desc',text='Description')
            tv.column('ID',width=50,anchor='center'); tv.column('Desc',width=150)
            for i,desc in d.items(): tv.insert('',tk.END,values=[i,desc])
            tv.pack(fill=tk.BOTH,expand=True)

    def on_click(self,event,field):
        x=round(event.x/self.pitch_img.width()*120,2)
        y=round(event.y/self.pitch_img.height()*80,2)
        w=self.widgets.get(field)
        if hasattr(w,'delete'): w.delete(0,tk.END); w.insert(0,f"{x},{y}")

    def start_timer(self):
        if not self.timer_running:
            self.timer_running=True; self.start_time=time.time()-self.elapsed_time; self.update_timer()
    def stop_timer(self):
        if self.timer_running:
            self.timer_running=False; self.elapsed_time=time.time()-self.start_time
            w=self.widgets['possession_duration']; w.delete(0,tk.END); w.insert(0,str(round(self.elapsed_time,2)))
    def reset_timer(self):
        self.timer_running=False; self.elapsed_time=0; self.timer_label.config(text='Duration: 0.00s')

    def update_timer(self):
        if self.timer_running:
            self.elapsed_time=time.time()-self.start_time
            self.timer_label.config(text=f'Duration: {round(self.elapsed_time,2)}s')
            self.after(100,self.update_timer)

    def add_record(self):
        try:
            rec = {}
            for k in columns:
                w = self.widgets.get(k)
                if isinstance(w, tk.BooleanVar):
                    rec[k] = w.get()
                else:
                    v = w.get()
                    rec[k] = parse_xy(v) if k.endswith('_location') else (
                        v if k in ['match', 'period'] else (float(v) if '.' in v else int(v)))
            # Append record
            self.df = pd.concat([self.df, pd.DataFrame([rec])], ignore_index=True)
            self.tree.insert('', tk.END, values=list(rec.values()))
            # Confirmation
            messagebox.showinfo('Success', 'Event added successfully!')
            # Clear fields
            for key, widget in self.widgets.items():
                if key == "match":
                    continue
                if isinstance(widget, tk.BooleanVar):
                    widget.set(False)
                else:
                    widget.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to add: {e}')
            messagebox.showerror('Error',f'Failed to add: {e}')

    def load_table(self):
        for _,r in self.df.iterrows(): self.tree.insert('',tk.END,values=list(r))
    def save_csv(self):
        self.df.to_csv(data_file,index=False); messagebox.showinfo('Saved','Events saved!')
    def on_close(self):
        if messagebox.askokcancel('Quit','Close application?'):
            self.destroy()

if __name__=='__main__':
  MatchRecorder().mainloop()
