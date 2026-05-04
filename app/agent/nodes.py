from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import LLM_Client
from app.agent.state import AgentState


class AgentNodes:

    @staticmethod
    def _invoke_with_retry(state: AgentState, prompt: str) -> tuple[str, str | None]:
        """
        Private helper. Tries to call LLM up to 3 times.
        Returns (response_text, error_message)
        If success → (text, None)
        If all retries fail → ("", error_message)
        """
        llm = LLM_Client.get()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                return response.content.strip(), None
            except Exception as e:
                retry_count = attempt + 1
                print(f"[retry] attempt {retry_count} failed: {str(e)}")

                if retry_count >= max_retries:
                    return "", f"LLM failed after {max_retries} attempts: {str(e)}"

        return "", "Unknown error"

    @staticmethod
    def classify_node(state: AgentState) -> dict:
        last_message = state["messages"][-1].content

        lowered = last_message.lower()

        prompt = f"""
                        Classify the user's message as exactly one word:
                        - question: asks for information or explanation
                        - acknowledge: is a personal statement or sharing something
                        - joke: asks for humor, a joke, something funny, or to make them laugh

                        Message: {lowered}

                        Reply with only one of these words:
                        question
                        acknowledge
                        joke
                        """

        response, error = AgentNodes._invoke_with_retry(state, prompt)
        if error:
            return {"error": error, "retry_count": state.get("retry_count", 0) + 1}
        intent = response.lower()
        print(f"[classify_node] intent: {intent}")
        return {"intent": intent}

    @staticmethod
    def answer_node(state: AgentState) -> dict:
        last_message = state["messages"][-1].content
        prompt = f"Answer this question helpfully: {last_message}"

        query = state.get("query", last_message)
        chunks = state.get("chunks", [])
        
        history = ""
        messages = state.get("messages", [])
        if len(messages) > 1:  # more than just current message
            history_messages = messages[:-1]  # all except current
            history_lines = []
            for msg in history_messages[-6:]:  # last 6 messages for context
                role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                history_lines.append(f"{role}: {msg.content}")
        history = "\n".join(history_lines)

        if not state.get("has_enough_context", False) or not chunks:
            prompt = f"""You are a helpful assistant. Answer based on the conversation history below.
                        If you don't know the answer, say so honestly.

                        Conversation history:
                        {history if history else "No previous conversation."}

                        Current question: {query}

                        Answer:"""
        response, error = AgentNodes._invoke_with_retry(state, prompt)
        if error:
            return {"error": error, "retry_count": state.get("retry_count", 0) + 1}
        return {
            "answer": response,
            "messages": [AIMessage(content=response)],
            "retry_count": 0,
            "error": None
        }
        
        # build context from chunks
        context = "\n\n".join([
            f"Source: {c['metadata'].get('source', 'unknown')} | Page: {c['metadata'].get('page', '?')}\n{c['content']}"
            for c in chunks
        ])
        
        prompt = f"""You are a helpful assistant. Answer the question using the document context provided.
                    Also consider the conversation history for context.

                    Conversation history:
                    {history if history else "No previous conversation."}

                    Document context:
                    {context}

                    Current question: {query}

                    Answer (cite sources when using document context):"""
        
        response, error = AgentNodes._invoke_with_retry(state, prompt)

        if error:
            return {"error": error, "retry_count": state.get("retry_count", 0) + 1}

        print(f"[answer_node] answer: {response[:60]}...")
        return {
            "answer": response,
            "messages": [AIMessage(content=response)],
            "error": None,
            "retry_count": 0,
        }

    @staticmethod
    def acknowledge_node(state: AgentState) -> dict:
        last_message = state["messages"][-1].content
        prompt = f"Respond empathetically to this statement: {last_message}"

        response, error = AgentNodes._invoke_with_retry(state, prompt)

        if error:
            return {"error": error, "retry_count": state.get("retry_count", 0) + 1}

        print(f"[acknowledge_node] response: {response[:60]}...")
        return {
            "answer": response,
            "messages": [AIMessage(content=response)],
            "retry_count": 0,
            "error": None,
        }

    @staticmethod
    def joke_node(state: AgentState) -> dict:
        last_message = state["messages"][-1].content
        prompt = f"Tell me a joke about: {last_message}"

        response, error = AgentNodes._invoke_with_retry(state, prompt)

        if error:
            return {"error": error, "retry_count": state.get("retry_count", 0) + 1}

        print(f"[joke_node] response: {response[:60]}...")
        return {
            "answer": response,
            "messages": [AIMessage(content=response)],
            "retry_count": 0,
            "error": None,
        }

    @staticmethod
    def error_node(state: AgentState) -> dict:
        error_message = state.get("error", "Something went wrong")
        print(f"[error_node] handling error: {error_message}")

        fallback = "I'm sorry, I'm having trouble processing your request right now. Please try again later."
        return {
            "answer": fallback,
            "messages": [AIMessage(content=fallback)],
            "retry_count": 0,
            "error": None,
        }

    @staticmethod
    def extract_query_node(state: AgentState) -> dict:
        """Extracts the plain query string from messages and stores in state."""
        query = state["messages"][-1].content
        print(f"[extract_query_node] query: {query}")
        return {"query": query}

    @staticmethod
    def retrieve_node(state: AgentState) -> dict:

        from app.rag.retriever import Retriever

        """Retrieves relevant chunks from vector store based on query."""

        query = state.get("query", state["messages"][-1].content)
        print(f"[retrieve_node] retrieving context for: '{query}'")

        try:
            retriever = Retriever()
            retriever.index()  # ensure indexed
            chunks = retriever.retrieve(query, k=4)

            if not chunks:
                return {"chunks": [], "has_enough_context": False, "error": None}

            relevant_chunks = [c for c in chunks if c["distance"] < 1.5]
            has_context = len(relevant_chunks) > 0
            return {
                "chunks": relevant_chunks,
                "has_enough_context": has_context,
                "error": None,
            }
        except Exception as e:
            return {
                "chunks": [],
                "has_enough_context": False,
                "error": f"Retrieval error: {str(e)}",
                "retry_count": state.get("retry_count", 0) + 1,
            }