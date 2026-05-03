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
            return {"error": error, "retry_count": state["retry_count"] + 1}
        intent = response.lower()
        print(f"[classify_node] intent: {intent}")
        return {"intent": intent}

    @staticmethod
    def answer_node(state: AgentState) -> dict:
        last_message = state["messages"][-1].content
        prompt = f"Answer this question helpfully: {last_message}"

        query = state["query"]
        chunks = state["chunks"]

        if not state["has_enough_context"] or not chunks:
            fallback = "I don't have enough information in my documents to answer that question."
            return {
                "answer": fallback,
                "messages": [AIMessage(content=fallback)],
                "retry_count": 0,
                "error": None,
            }
        
        # build context from chunks
        context = "\n\n".join([
            f"Source: {c['metadata'].get('source', 'unknown')} | Page: {c['metadata'].get('page', '?')}\n{c['content']}"
            for c in chunks
        ])
        
        prompt = f"""Answer the question using ONLY the context provided below.
            If the answer is not in the context, say "I don't have enough information."
            Always mention which source and page your answer comes from.

            Context:
            {context}

            Question: {query}

            Answer:"""
        
        response, error = AgentNodes._invoke_with_retry(state, prompt)

        if error:
            return {"error": error, "retry_count": state["retry_count"] + 1}

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
            return {"error": error, "retry_count": state["retry_count"] + 1}

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
            return {"error": error, "retry_count": state["retry_count"] + 1}

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

        query = state["query"]
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
                "retry_count": state["retry_count"] + 1,
            }