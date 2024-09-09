
### LLM Story Game

This interactive game challenges you to survive by keeping your HP above 0. You'll use mana to select the right actions, while the outcome of future events is influenced by your luck. Each event is visually represented by images generated using the Flux [dev] model with warm inference via Hugging Face.

To start playing, simply provide your OpenAI GPT API key and HuggingFace token in an `.env` file, as the game performs best when integrated with GPT models.

#### Features:
- **HP, Mana, and Luck system**: Manage your resources carefully to survive.
- **Dynamic event images**: Flux [dev] generates event images in real-time.
- **Luck-driven events**: Unpredictable outcomes based on your luck.
------
#### Architecture:
I used mainly langchain chains and well curated prompts to get the output as needed, then the hp,mana and luck setup was pure python programming. Everything was wrapped up in a streamlit app.
![LLM_Game](https://github.com/user-attachments/assets/bd498292-82ff-451d-bf18-6a631deca3aa)

------
#### Limitation: 
A big limitation about this project is that it takes so much time to generate images, I can't use flux on my poor 6Gb VRAM GPU since it take at least 22Gb to run it locally.
If you're interested in turning this small project into a wider one (or a commercial venture, or maybe a project that resambles it), feel free to contact me

