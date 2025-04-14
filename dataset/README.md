# Reasoning Datasets

This README contains details on the structure of each dataset we study and a description of its purpose. 

## Datasets

### pdsv2_multiscore_sft

**Purpose**: Train on examples of highly rated stories (average PDS score 3.5+) to improve depth directly.

**Structure**:

```

"conversations": [
    {
    "from": "human",
    "value": "You are a seasoned writer who has won several accolades for your emotionally rich stories. \nWhen you write, you delve deep into the human psyche, pulling from the reservoir of universal \nexperiences that every reader, regardless of their background, can connect to.\nYour writing is renowned for painting vivid emotional landscapes, making readers not just observe \nbut truly feel the world of your characters.\nEvery piece you produce aims to draw readers in, encouraging them to reflect on their own lives \nand emotions.\nYour stories are a complex tapestry of relationships, emotions, and conflicts, \neach more intricate than the last.\n\n\nWrite a 500-word story on the following prompt:\nYou died and awoke in the afterlife. It's quite nice actually. The people and atmosphere are a lot nicer than you are used to and there is no stress or pressure. When you ask what good deed got you into heaven you are informed that this is hell, followed by a visit from a very concerned demon.\n\nOnly respond with the story."
    },
    {
    "from": "gpt",
    "value": "Susan Johnson looked around in awe as she stepped through the pearly gates. The stunning beauty of the heavenly landscape left her breathless. Warm, loving energy enveloped her, and she felt an overwhelming sense of belonging. She couldn't believe she made it to heaven. A kindly angel greeted her and showed her to her celestial mansion. Susan felt a mixture of joy and disbelief, thinking, \"I didn't do enough good deeds to deserve this.\"\n\nLater that day, Susan met Mark Davis at a heavenly picnic. He was the life of the party, telling jokes and entertaining the crowd. Susan noticed he had a more serious side when he was alone, though. He appeared to be calculating and stressed. Susan couldn't help but feel drawn to him, wondering if they both shared the same secret.\n\nOne evening, as Susan sat by a crystal clear river, she confided in Mark her fears about not being good enough for heaven. Mark opened up about his dark side, the manipulation and gossip that haunted him. They both wondered if this was hell instead, a twisted version of heaven designed to make them doubt themselves. Just then, a tall, imposing demon appeared before them.\n\n\"Hello, I am here to guide you,\" the demon said, its voice gentle yet commanding.\n\nSusan and Mark exchanged worried glances, but the demon reassured them, \"Do not worry, I am here to help.\"\n\nThe demon explained that Susan's selflessness and kindness were what brought her to heaven. As for Mark, his philanthropy and ability to bring joy to others outweighed his darker side. They were both where they belonged.\n\nTears welled up in Susan's eyes as she realized that her lifelong belief in her own unworthiness was a lie. Mark felt a heavy weight lift from his shoulders, knowing that even with his flaws, he could still find peace. The demon smiled, its concern for the souls it guided evident in its eyes.\n\nIn the days that followed, Susan and Mark worked together to spread the message that nobody is perfect, but with genuine intentions and actions, anyone can find salvation. They dedicated themselves to helping others understand that hell isn't a place of punishment but rather a chance for personal growth and redemption.\n\nAs Susan and Mark continued their mission, they found solace in knowing that even in the afterlife, forgiveness and understanding can change one's destiny. And the demon, with a renewed sense of purpose, continued to guide lost souls, hoping to show them the way to their true selves and ultimate happiness."
    }
]

```
Data Path: `./LLaMA-Factory/data/pdsv2/sft/pdsv2_multiscore_sft.json`
Formatter: `./dataset/formatters/format_pds.py`