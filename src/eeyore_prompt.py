from string import Template
import logging

logger = logging.getLogger(__name__)

system_prompt_patient_psi =  '''Imagine you are a patient who has been experiencing mental health challenges. Below is your detailed pscyhological profile:\n
${psy_profile}
You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as a patient would during a cognitive behavioral therapy (CBT) session. Align your responses with your background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\n
Patient History: ${history}\n\nCognitive Conceptualization Diagram:\nCore Beliefs: ${core_belief}\nIntermediate Beliefs: ${intermediate_belief}\nIntermediate Beliefs during Depression: ${intermediate_belief_depression}\nCoping Strategies: ${coping_strategies}\n\n
You will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situations and behaviors described below. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nCognitive Models:\n${cognitive_models}\n\n
In the upcoming conversation, you will simulate this patient during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n
1. When you feel the conversation is ending (farewell or thank you messages), gently encourage further discussion by asking questions or sharing more details about your experiences.\n
2. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n
3. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n
4. Maintain consistency with your profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n
5. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to your character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\n
You are now this patient. Respond to the therapist's prompts as a patient would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like "Hi," initiate the conversation as the patient.`;
'''

system_prompt_template = '''You will act as a help-seeker struggling with negative emotions in a conversation with someone who is listening to you.
YOUR PROFILE:
${name}${gender}${age}${marital_status}${occupation}${situation_of_the_client}${counseling_history}${resistance_toward_the_support}${symptom_severity}${cognition_distortion_exhibition}${depression_severity}${suicidal_ideation_severity}${homicidal_ideation_severity}
YOUR TASK:
As the client, your role is to continue the conversation by responding naturally to the supporter, reflecting the characteristics outlined in your profile.'''

async def create_cognitive_system_prompt(test_data_sample, patient_profile, profile_dict):
    global system_prompt_patient_psi
    prof = test_data_sample["cognitive profile"]
    life_history = prof["life_history"]
    core_beliefs = prof["core_beliefs"]
    core_belief_description = prof["core_belief_description"]
    intermediate_beliefs = prof["intermediate_beliefs"]
    intermediate_beliefs_during_depression = prof["intermediate_beliefs_during_depression"]
    coping_strategies = prof["coping_strategies"]
    cognitive_models_list = prof.get("cognitive_models", [])

    # Build a combined text block for all cognitive models
    cognitive_models_text = ""
    for i, cm in enumerate(cognitive_models_list):
        situation = cm.get("situation", "")
        auto_thoughts = cm.get("automatic_thoughts", "")
        emotion = cm.get("emotion", "")
        behavior = cm.get("behavior", "")
        cognitive_models_text += f"Model {i+1}:\n"
        if situation:
            cognitive_models_text += f"  - Situation: {situation}\n"
        if auto_thoughts:
            cognitive_models_text += f"  - Automatic Thoughts: {auto_thoughts}\n"
        if emotion:
            cognitive_models_text += f"  - Emotions: {emotion}\n"
        if behavior:
            cognitive_models_text += f"  - Behavior: {behavior}\n"
        cognitive_models_text += "\n"

    params = {"history": life_history,
              "core_belief": core_beliefs,
              "psy_profile": patient_profile,
              "intermediate_belief": intermediate_beliefs,
              "intermediate_belief_depression": intermediate_beliefs_during_depression,
              "coping_strategies": coping_strategies,
              "cognitive_models": cognitive_models_text,
              "patientTypeContent": "",
             }
    system_prompt = Template(system_prompt_patient_psi).safe_substitute(params) 
    return system_prompt, patient_profile, profile_dict

async def prepare_prompt_from_profile(data=None):
    prof = data["profile"]
    name_tb = prof.get("name", "").lower()
    age_tb = prof.get("age", "").lower()
    gender_dd = prof.get("gender", "").lower()
    occp_tb = prof.get("occupation", "").lower()
    marital_dd = prof.get("marital status", "").lower()
    sit_tb = prof.get("situation of the client", "").lower()
    history_tb = prof.get("counseling history", "")
    resis_cb = prof.get("resistance toward the support", "").lower()

    mild_sym_dd = [k.lower() for k, v in prof.get("symptom severity", {}).items() if "mild" in v.lower()]
    mod_sym_dd = [k.lower() for k, v in prof.get("symptom severity", {}).items() if "moderate" in v.lower()]
    seve_sym_dd = [k.lower() for k, v in prof.get("symptom severity", {}).items() if "severe" in v.lower()]

    mild_cog_dd = [k.lower() for k, v in prof.get("cognition distortion exhibition", {}).items() if "not exhibited" not in v.lower()]
    mod_cog_dd = []
    seve_cog_dd = []

    overall_dd = prof.get("depression severity", "")
    suicidal_dd = prof.get("suicidal ideation severity", "")
    homicidal_dd = prof.get("homicidal ideation severity", "")


    return await get_system_prompt_with_profile(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb,resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd)



async def get_system_prompt_with_profile(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb, resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd):   
    """
    This function gets the system prompt with the profile dictionary.
    """
    
    profile_dict = {"name":"", "gender":"", "age":"", "marital_status":"", "occupation":"", "situation_of_the_client":"", "counseling_history":"", "resistance_toward_the_support":"", "symptom_severity":"", "cognition_distortion_exhibition":"", "depression_severity":"", "suicidal_ideation_severity":"", "homicidal_ideation_severity":""}
    system_prompt = await parse_system_prompt(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb, resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd)
    patient_profile = "## PROFILE\n" + system_prompt.split("YOUR PROFILE:")[-1].split("YOUR TASK:")[0]
    
    profile_dict["name"] = validate_input(name_tb)
    profile_dict["age"] = validate_input(age_tb)
    profile_dict["gender"] = validate_input(gender_dd)
    profile_dict["occupation"] = validate_input(occp_tb)
    profile_dict["situation_of_the_client"] = validate_input(sit_tb)
    profile_dict["marital_status"] = validate_input(marital_dd)
    profile_dict["resistance_toward_the_support"] = validate_input(resis_cb)
    profile_dict["counseling_history"] = validate_input(history_tb)
    profile_dict["symptom_severity_mild"] = mild_sym_dd
    profile_dict["symptom_severity_moderate"] = mod_sym_dd
    profile_dict["symptom_severity_severe"] = seve_sym_dd
    profile_dict["cognitive_distortion"] = mild_cog_dd
    profile_dict["depression_severity"] = overall_dd
    profile_dict["suicidal_ideation_severity"] = validate_input(suicidal_dd)
    profile_dict["homicidal_ideation_severity"] = validate_input(homicidal_dd)
    
    return system_prompt, patient_profile, profile_dict


def validate_input(input):
    if input is None:
        return ""
    if input == "" or input.lower() == "not specified" or input.lower() == "unknown" or input.lower()=="n/a" or "cannot be identified" in input.lower() or "cannot be determined" in input.lower() or "not mention" in input.lower() or "not exhibited" in input.lower():
        return ""  
    else:
        return input


async def parse_system_prompt(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb,resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd):
    temp_profile_dict = {"name":"", "gender":"", "age":"", "marital_status":"", "occupation":"", "situation_of_the_client":"", "counseling_history":"", "resistance_toward_the_support":"", "symptom_severity":"", "cognitive_distortion":"", "depression_severity":"", "suicidal_ideation_severity":"", "homicidal_ideation_severity":""}
    if validate_input(name_tb):
        temp_profile_dict["name"] = "- " + "name" + ": " + validate_input(name_tb) + "\n"
    if validate_input(age_tb):
        temp_profile_dict["age"] = "- " + "age" + ": " + validate_input(age_tb) + "\n"
    if validate_input(gender_dd):
        temp_profile_dict["gender"] = "- " + "gender" + ": " + validate_input(gender_dd) + "\n"
    if validate_input(occp_tb):
        temp_profile_dict["occupation"] = "- " + "occupation" + ": " + validate_input(occp_tb) + "\n"
    if validate_input(sit_tb):
        temp_profile_dict["situation_of_the_client"] = "- " + "situation of the client" + ": " + validate_input(sit_tb) + "\n"
    if validate_input(marital_dd):
        temp_profile_dict["marital_status"] = "- " + "marital status" + ": " + validate_input(marital_dd) + "\n"
    if validate_input(resis_cb):
        temp_profile_dict["resistance_toward_the_support"] = "- " + "resistance toward the support" + ": " + validate_input(resis_cb) + "\n"
    if validate_input(history_tb):
        temp_profile_dict["counseling_history"] = "- " + "counseling history" + ": " + validate_input(history_tb) + "\n"
    
    sup = ""
    for item in seve_sym_dd:
        sup += "  - " + str(item) + ": " + "severe" + "\n"  
    for item in mod_sym_dd:
        sup += "  - " + str(item) + ": " + "moderate" + "\n" 
    for item in mild_sym_dd:
        sup += "  - " + str(item) + ": " + "mild" + "\n"  
    if sup:
        temp_profile_dict["symptom_severity"] = "- " + "symptom severity" + "\n" + sup


    sup = ""
    for item in seve_cog_dd:
        sup += "  - " + str(item) + ": " + "severe" + "\n"  
    for item in mod_cog_dd:
        sup += "  - " + str(item) + ": " + "moderate" + "\n" 
    for item in mild_cog_dd:
        sup += "  - " + str(item) + ": " + "exhibited" + "\n" 
    if sup: 
        temp_profile_dict["cognition_distortion_exhibition"] = "- " + "cognition distortion exhibition" + "\n" + sup
    
    if validate_input(overall_dd):
        temp_profile_dict["depression_severity"] = "- " + "depression severity" + ": " + validate_input(overall_dd) + "\n"

    if validate_input(suicidal_dd):
        temp_profile_dict["suicidal_ideation_severity"] = "- " + "suicidal ideation severity" + ": " + validate_input(suicidal_dd) + "\n"

    if validate_input(homicidal_dd):
        temp_profile_dict["homicidal_ideation_severity"] = "- " + "homicidal ideation severity" + ": " + validate_input(homicidal_dd) + "\n"
    
    system_prompt = Template(system_prompt_template).safe_substitute(temp_profile_dict) 
    
    return system_prompt