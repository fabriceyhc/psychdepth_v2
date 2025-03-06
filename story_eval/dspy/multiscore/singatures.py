import dspy

class PDSMultiScore(dspy.Signature):
    """
    1. Review the given components of psychological depth: authenticity, emotion provoking, 
       empathy, engagement, and narrative complexity. Be sure to understand each concept and 
       the questions that characterize them.
    2. Read a given story, paying special attention to components of psychological depth.
    3. Assign a rating for each component from 1 to 5. 
       (1 = greatly below average, 3 = average, 5 = greatly above average).
    4. Lastly, estimate the likelihood that each story was authored by a human or an LLM. 
       (1 = very likely LLM, 5 = very likely human).

    ### Description of Psychological Depth Components:
    
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
        - Do you feel that the text led you to introspection, or to new insights about yourself or the world?
    - Engagement 
        - Does the text engage you on an emotional and psychological level?
        - Do you feel the need to keep reading as you read the text?
    - Narrative Complexity 
        - Do the characters in the story have multifaceted personalities? 
          Are they developed beyond stereotypes or tropes? Do they exhibit internal conflicts?
        - Does the writing explore the complexities of relationships between characters?
        - Does it delve into the intricacies of conflicts and their partial or complete resolutions?
    """
    
    story: str = dspy.InputField(desc="A story to be evaluated for the different components of psychological depth.")
    
    authenticity_score: float = dspy.OutputField(desc="1=Unrealistic, 5=Profoundly authentic")
    emotion_provoking_score: float = dspy.OutputField(desc="1=Flat, 5=Deeply moving")
    empathy_score: float = dspy.OutputField(desc="1=Detached, 5=Transformative empathy")
    engagement_score: float = dspy.OutputField(desc="1=Boring, 5=Irresistibly compelling")
    narrative_complexity_score: float = dspy.OutputField(desc="1=Shallow, 5=Masterfully complex")
    human_likeness_score: float = dspy.OutputField(desc="1=Clearly AI, 5=Undeniably human")