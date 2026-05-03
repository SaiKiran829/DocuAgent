from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import AgentNodes
from app.agent.router import AgentRouter

class AgentGraph:
    _graph = None

    @classmethod
    def build(cls) -> StateGraph:
        if cls._graph is not None:
            return cls._graph
        
        graph = StateGraph(AgentState)

        graph.add_node("classify", AgentNodes.classify_node)
        graph.add_node("answer", AgentNodes.answer_node)
        graph.add_node("acknowledge", AgentNodes.acknowledge_node)
        graph.add_node("joke", AgentNodes.joke_node)
        graph.add_node("error", AgentNodes.error_node)
        graph.add_node("extract_query", AgentNodes.extract_query_node)
        graph.add_node("retrieve", AgentNodes.retrieve_node)

        #entry point
        graph.set_entry_point("extract_query")
        
        graph.add_edge("extract_query", "classify")

        #conditional edges from classify
        graph.add_conditional_edges(
            "classify",
            AgentRouter.route_by_intent,
            {
                "answer": "retrieve",
                "acknowledge": "acknowledge",
                "joke": "joke",
                "error": "error"
            }
        )
        
        graph.add_conditional_edges(
            'answer',
            AgentRouter.route_after_answer,
            {
                "end": END,
                "error": "error"
            }
        )
        
        graph.add_conditional_edges(
            'acknowledge',
            AgentRouter.route_after_answer,
            {
                "end": END,
                "error": "error"
            }
        )
        
        graph.add_conditional_edges(
            'joke',
            AgentRouter.route_after_answer,
            {
                "end": END,
                "error": "error"
            }
        )
        
        graph.add_edge("retrieve", "answer")
        
        

        graph.add_edge("error", END)
        cls._graph = graph.compile()
        return cls._graph