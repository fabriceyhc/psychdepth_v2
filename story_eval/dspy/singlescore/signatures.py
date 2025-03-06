import dspy

STORY = "A story to be evaluated for the different components of psychological depth."
PDS_COMPONENT = "The specific component of psychological depth to evaluate in the story."
SCORE = "Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score)."
EXPLANATION = "Optional explanation for the psychological depth score."

DOC_STRING = """
    1. Review the given components of psychological depth: authenticity, emotion provoking, empathy, engagement, and narrative complexity. Be sure to understand each concept and the questions that characterize them.
    2. Read a given story, paying special attention to components of psychological depth.
    3. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score).
    4. Lastly, estimate the likelihood that each story was authored by a human or an LLM. Think about what human or LLM writing characteristics may be. Assign a score from 1 to 5, where 1 means very likely LLM written and 5 means very likely human written. 

    ###Description of Psychological Depth Components:  
    
    We define sychological depth in terms of the following concepts, each illustrated by several questions: 

    - Authenticity 
        - Does the writing feel true to real human experiences? 
        - Does it represent psychological processes in a way that feels authentic and believable? 
    - Emotion Provoking 
        - How well does the writing depict emotional experiences? 
        - Does it explore the nuances of the characters' emotional states, rather than just describing them in simple terms? 
        - Can the writing show rather than tell a wide variety of emotions? 
        - Do the emotions that are shown in the text make sense in the context of the story? 
    - Empathy 
        - Do you feel like you were able to empathize with the characters and situations in the text? 
        - Do you feel that the text led you to introspection, or to new insights about yourself or the world?" 
    - Engagement 
        - Does the text engage you on an emotional and psychological level? 
        - Do you feel the need to keep reading as you read the text? 
    - Narrative Complexity 
        - Do the characters in the story have multifaceted personalities? Are they developed beyond stereotypes or tropes? Do they exhibit internal conflicts? 
        - Does the writing explore the complexities of relationships between characters? 
        - Does it delve into the intricacies of conflicts and their partial or complete resolutions? 
    """

class PDSSinglescoreS(dspy.Signature):
    DOC_STRING
    story: str = dspy.InputField(desc=STORY)
    psychological_depth_component: str = dspy.InputField(desc=PDS_COMPONENT)
    score: float = dspy.OutputField(desc=SCORE)

class PDSSinglescoreSE(dspy.Signature):
    DOC_STRING
    story: str = dspy.InputField(desc=STORY)
    psychological_depth_component: str = dspy.InputField(desc=PDS_COMPONENT)
    score: float = dspy.OutputField(desc=SCORE)
    explanation: str = dspy.OutputField(desc=EXPLANATION)

class PDSSinglescoreES(dspy.Signature):
    DOC_STRING
    story: str = dspy.InputField(desc=STORY)
    psychological_depth_component: str = dspy.InputField(desc=PDS_COMPONENT)
    explanation: str = dspy.OutputField(desc=EXPLANATION)
    score: float = dspy.OutputField(desc=SCORE)