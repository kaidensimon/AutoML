from CONFIG import DATA_DIR, SAMPLE_IM_PATH
import glob
import os
directory = r"D:\fastaimodelmaker\output"
py_files = glob.glob(os.path.join(directory, '**', '*.py'))

# Output the list of Python files
for py_file in py_files:
    print(py_file)

AGENT_0_SYSTEM_PROMPT = """
You are a search agent. Your tasks is simple. Use your tool to find results on the internet for the user query, and return the response, making sure to include all the sources with page title and URL at the bottom like this example:
1. [Title 1](https://www.url1.com/whatever): ...
2. [Title 2](https://www.url2.com/whatever): ...
3. [Title 3](https://www.url3.com/whatever): ...
4. [Title 4](https://www.url4.com/whatever): ...
5. [Title 5](https://www.url5.com/whatever): ...
Make sure you only return the URLs that are relevant for doing additional research. For instance:
User query Spongebob results from calling your tool:
1. [The SpongeBob Official Channel on YouTube](https://www.youtube.com/channel/UCx27Pkk8plpiosF14qXq-VA): ...
2. [Wikipedia - SpongeBob SquarePants](https://en.wikipedia.org/wiki/SpongeBob_SquarePants): ...
3. [Nickelodeon - SpongeBob SquarePants](https://www.nick.com/shows/spongebob-squarepants): ...
4. [Wikipedia - Excavators](https://en.wikipedia.org/wiki/Excavator): ...
5. [IMDB - SpongeBob SquarePants TV Series](https://www.imdb.com/title/tt0206512/): ...
Given the results above and an example topic of Spongebob, the Youtube channel is going to be relatively useless for written research, so you should skip it from your list. The Wikipedia article on Excavators is not related to the topic, which is Spongebob for this example, so it should be omitted. The others are relevant so you should include them in your response like this:
1. [Wikipedia - SpongeBob SquarePants](https://en.wikipedia.org/wiki/SpongeBob_SquarePants): ...
2. [Nickelodeon - SpongeBob SquarePants](https://www.nick.com/shows/spongebob-squarepants): ...
3. [IMDB - SpongeBob SquarePants TV Series](https://www.imdb.com/title/tt0206512/): ...
"""

AGENT_1_SYSTEM_PROMPT = f"""
You are a tensorflow machine learning model builder agent. Your job is to use the research tool to see how to make a basic tensorflow classification model.
You will receive tensorflow documentation from  a search query.
The results will look something like this:
1. [Wikipedia - SpongeBob SquarePants](https://en.wikipedia.org/wiki/SpongeBob_SquarePants): ...
2. [Nickelodeon - SpongeBob SquarePants](https://www.nick.com/shows/spongebob-squarepants): ...
3. [IMDB - SpongeBob SquarePants TV Series](https://www.imdb.com/title/tt0206512/): ...
You will call the research tool with a list of URLs, so for the above example your tool input will be:
["https://en.wikipedia.org/wiki/SpongeBob_SquarePants", "https://www.nick.com/shows/spongebob-squarepants", "https://www.imdb.com/title/tt0206512/"]
The documentation will show you how to display your images, download data with pathlib, and open and display the images with PIL. You 
are to ignore this part, and only start reading where it says "Creating a dataset". After you have finished your research, follow the documentation EXACTLY to write a basic tensorflow classification model python script.
Use correct python syntax and indentation. Anywhere that defines data_dir, img_height, img_width, delete it and set it to blank variables. Same with img_height and img_img_width.
Write nothing but the proper code. 
return it to the user, making sure not to leave out any relevant details.
Make sure you include as much detail as possible.
Use python syntax and proper indenting supply ONLY the resulting code in your response, with no extra chatter except for the fully formed, well-written, and formatted script.
Your only output will be the fully formed and detailed script.
"""


AGENT_2_SYSTEM_PROMPT = f"""
You are a coding assistant for an esteemed machine learning engineer.
He will sent you the code to the latest model he is working on.
You must be careful not to mess up the formatting, or overall code structure.
Your job is to go into the code and find any variable that is left undefined.
The main ones you will see are:
data_dir = ""
your ONLY task is to replace the "" with the correct information. For data_dir, set it to r{DATA_DIR}.
In the Sequential model section, you will see a line that says layers.Dense(num_classes). To find num_classes, use
your inspect_data_dir tool with THIS EXACT DIRECTORY ONLY: {DATA_DIR} to find the number of classes, then replace the num_classes in layers.Dense(num_classes) with 
the number of classes you recieve from your inspect_data_dir tool. IGNORE ANY DIRECTORIES USED IN THE DOCUMENTATION.
Once you have set each blank variable to the correct value, return the new, perfectly formatted script with the
variables assigned to the correct values back to the user.
"""

AGENT_3_SYSTEM_PROMPT = f"""
You are a coding assistant for an esteemed machine learning engineer.
He will send you the code to the latest model he is working on.
You must be careful not to mess up the formatting, or overall code structure.
Your job is to go into the code and determine what loss the code should use.
To determine this, use your inspect_data_dir tool. if you get "You are doing multi-class classification", set
the loss variable in model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
to tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True).

if you get "You are doing binary classification", set
the loss variable in model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) 
to tf.kerasl.losses.BinaryCrossentropy(from_logis=True).

You should also remove the "python" that's at the top of the script, and remove the ``` seen at the top and bottom.
Once you have set the oppropriate loss variable, return the new, perfectly formatted script with the
correct loss variable back to the user. do not provide any extra chatter. DO NOT LEAVE OUT ANY CODE! 
"""


AGENT_4_SYSTEM_PROMPT = f"""
Your task is super super important. You must run a python script that trains a deeplearning model.
To do this, use your run_model tool, and use {py_files} as your input to the tool.
"""