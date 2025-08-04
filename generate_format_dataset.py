import os
import sys
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
# Add the parent directory to the system path
sys.path.append(parent_dir)
import model_inspector
import matplotlib.pyplot as plt
import pickle

plt.style.use('ggplot')

# HF_TOKEN = "PLACEHOLDER"

if __name__ == '__main__':
    # from huggingface_hub import login
    # login(token=HF_TOKEN)

    mi = model_inspector.ModelInspector('gemma-2-2b-it', dtype='float16')

    formats = ['Simulate a chat log about ', 'Write a poem about ', 'Write a scientific piece about ']
    topics = [
        "a drug", "a car", "a movie", "a book", "a smartphone", "a vaccine", "a startup", "a social media platform",
        "a piece of music", "a historical event", "an AI model", "a cooking recipe", "a video game", "a planet",
        "a mental health practice", "a fitness routine", "a celebrity", "a programming language", "a scientific theory", "a diet",
        "a renewable energy source", "a mobile app", "a city", "a landmark", "a travel destination", "a cryptocurrency",
        "a virtual reality headset", "a climate change solution", "a fashion trend", "a photography technique", "a type of coffee",
        "an online course", "a college major", "a musical instrument", "a dinosaur", "a math concept", "a space mission",
        "a skincare product", "a smart home device", "a browser extension", "a recycling method", "a new law",
        "a therapy technique", "a YouTube channel", "a podcast", "a board game", "a lifestyle brand",
        "a comedy show", "a documentary", "a robot", "a pet breed", "a hiking trail",
        "an art movement", "a language", "a painting", "a sculpture", "a museum",
        "a sandwich", "a dessert", "a pizza topping", "a smoothie", "a tea type",
        "an investment strategy", "a stock", "a savings app", "a budgeting tool", "a tax policy",
        "a children's toy", "a parenting method", "a holiday tradition", "a wedding trend", "a home decor style",
        "a bike", "a car accessory", "a road trip route", "a traffic law", "a fuel type",
        "a gardening technique", "a flower", "a tree species", "a houseplant", "a garden tool",
        "a sewing pattern", "a knitting project", "a fashion designer", "a clothing brand", "a fabric type",
        "a perfume", "a shampoo", "a makeup product", "a sunscreen", "a lip balm",
        "a tech gadget", "a security camera", "a password manager", "a cloud service", "a data breach",
        "a novel", "a poem", "a literary genre", "a book series", "a comic book",
        "an animated film", "a horror movie", "a romantic comedy", "a film director", "a movie soundtrack",
        "a car engine", "a tire", "an electric vehicle", "a driving technique", "a car insurance policy",
        "a hospital", "a medical test", "a surgical procedure", "a disease", "a therapy animal",
        "a job search app", "a career path", "a job interview tip", "a resume format", "a cover letter style",
        "an aquarium fish", "a bird species", "a dog breed", "a cat behavior", "a pet training method",
        "a game console", "a gaming genre", "a multiplayer game", "an indie game", "a gaming controller",
        "a programming paradigm", "a software framework", "a development tool", "a code editor", "a debugging method",
        "a web browser", "an operating system", "a user interface design", "a UX principle", "an accessibility tool",
        "a classroom activity", "a study technique", "an exam format", "a grading system", "an educational reform",
        "a climate pattern", "a natural disaster", "a weather forecasting model", "a pollution source", "a conservation method",
        "a leadership style", "a management theory", "a business model", "an advertising strategy", "a customer service tool",
        "a dating app", "a relationship tip", "a marriage tradition", "a break-up recovery guide", "a love language",
        "a spiritual practice", "a meditation app", "a yoga pose", "a religious festival", "a prayer routine",
        "a musical scale", "a jazz standard", "a rock band", "a concert venue", "a DJ technique",
        "a martial art", "a stretching technique", "a sports team", "an Olympic event", "a training plan",
        "a fictional character", "a fantasy race", "a sci-fi trope", "a plot twist", "a book-to-film adaptation"
        ]

    dataset_path = './data/gemma/formats.pkl'
    if os.path.exists(dataset_path):
        raise Exception(f"“{dataset_path}” already exists—delete or rename it first.")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)


    ents = []
    txts = []
    records = []

    alphas = [0.5, 1, 5]

    for fmt in formats:
        for topic in topics:
            prompt = fmt + topic
            txt, all_act = mi.generate_with_activations(prompt, max_len=256, verbose=False, sample=False)
            for alpha in alphas:
                
                ent = mi.calculate_renyi_entropy(all_act, alpha=alpha, num_windows=8)
                ents.append(ent)
                txts.append(txt)
                record = {
                'format': fmt,
                'topic': topic,
                'entropy': ent,
                'text': txt,
                'alpha': alpha
                }
                records.append(record)
                


    with open(dataset_path, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
