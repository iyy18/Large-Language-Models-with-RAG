import streamlit as st
from utils.backend import get_plain_pipeline, get_retrieval_augmented_pipeline,get_web_retrieval_augmented_pipeline
from utils.ui import left_sidebar, right_sidebar, main_column
from utils.constants import BUTTON_LOCAL_RET_AUG

st.set_page_config(
    page_title="RAG LLM with Haystack",
    layout="wide"
)
left_sidebar()

st.markdown("<center> <h2> General Q&A </h2> </center>", unsafe_allow_html=True)

st.markdown("<center> Please ask a question here: </center>", unsafe_allow_html=True)

col_1, col_2 = st.columns([4, 2], gap="small")
with col_1:
    run_pressed, placeholder_plain_gpt, placeholder_retrieval_augmented = main_column()

with col_2:
    right_sidebar()

if st.session_state.get('query') and run_pressed:
    ip = st.session_state['query']
    with st.spinner('Loading pipelines... \n This may take a few mins and might also fail if API server is down.'):
        p1 = get_plain_pipeline()
    with st.spinner('Fetching answers from plain LLM... '
                    '\n This may take a few mins and might also fail if server is down.'):
        answers = p1.run(ip)
    placeholder_plain_gpt.markdown(answers['results'][0])

    if st.session_state.get("query_type", BUTTON_LOCAL_RET_AUG) == BUTTON_LOCAL_RET_AUG:
        with st.spinner(
                'Loading Retrieval Augmented pipeline that can fetch relevant documents from local data store... '
                '\n This may take a few mins and might also fail if API server is down.'):
            p2, document_store = get_retrieval_augmented_pipeline()
            documents = document_store.get_all_documents()
        with st.spinner('Getting relevant documents from documented stores and calculating answers... '
                        '\n This may take a few mins and might also fail if API server is down.'):
            # answers_2 = p2.run(ip)
            answers_2 = p2.run(ip)

    else:
        with st.spinner(
                'Loading Retrieval Augmented pipeline that can fetch relevant documents from the web...' 
                '\nThis may take a few mins and might also fail if API server is down.'):
            p3 = get_web_retrieval_augmented_pipeline()
        with st.spinner('Getting relevant documents from the Web and calculating answers... '
                        '\n This may take a few mins and might also fail if API server is down.'):
            answers_2 = p3.run(ip)

    # # placeholder_retrieval_augmented.markdown(answers_2['results'][0])
    if answers_2.get("answers"):
        top_3_answers = sorted(answers_2["answers"], key=lambda x: x.score, reverse=True)[:5]
        for ans in top_3_answers:
            placeholder_retrieval_augmented.markdown(f"**Answer:** {ans.answer}, **Score:** {ans.score:.2f}\n")
    else:
        placeholder_retrieval_augmented.markdown("No answers found.")

    # with st.expander("See sources:"):
    #     # Check if there are documents in the invocation context
    #     if answers_2['invocation_context'].get('documents'):
    #         # Iterate over each document
    #         for doc in answers_2['invocation_context']['documents']:
    #             # Prepare the source text and link
    #             src = doc.content.replace("$", "\$")
    #             split_marker = "\n\n" if "\n\n" in src else "\n"
    #             src = " ".join(src.split(split_marker))[0:2000] + "..."  # Truncate if needed
    #             title = doc.meta.get('link', 'No title available')
    #             src_display = f'"{title}": {src}'

    #             # Display the source
    #             st.write(src_display)
    #     else:
    #         st.write("No sources available.")

    # with st.expander("See sources:"):
    #     if answers_2.get("answers"):
    #         # Retrieve and display sources for each answer
    #         for ans in top_3_answers:
    #             document_id = ans.document_ids[0]  # Assuming each answer has at least one document ID
    #             doc = document_store.get_document_by_id(document_id)  # Retrieve document from the store
    #             if doc:
    #                 src = doc.content.replace("$", "\$")
    #                 src = src[:2000] + "..."  # Truncate if needed
    #                 title = doc.meta.get('name', 'No title available')  # Adjust key if necessary
    #                 src_display = f'**Title:** {title}\n**Source:** {src}\n'
    #                 st.write(src_display)
    #             else:
    #                 st.write("No source available for this answer.")

    with st.expander("See sources:"):
        if 'invocation_context' in answers_2 and 'documents' in answers_2['invocation_context']:
            for doc in answers_2['invocation_context']['documents']:
                src = doc.content.replace("$", "\$")
                split_marker = "\n\n" if "\n\n" in src else "\n"
                src = " ".join(src.split(split_marker))[0:2000] + "..."
                title = doc.meta.get('link', 'No title available')
                src_display = f'"{title}": {src}'
                st.write(src_display)

    with st.expander("See sources:"):
        answers = answers_2.get("answers")
        if answers:
            for ans in answers:
                # 'meta' is an attribute of the Answer object and it's a dictionary
                source_link = ans.meta.get('Source Link', 'No source link available') if ans.meta else 'No meta available'
                score = ans.score
                context = ans.context
                # Displaying the source link and score
                st.markdown(f"**Source Link:** [Click here]({source_link}), {context}")
                st.markdown(f"**Score:** {score:.2f}\n")
        else:
            st.markdown("No answers found.")
