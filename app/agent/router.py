from app.agent.state import AgentState

class AgentRouter:

    @staticmethod
    def route_by_intent(state: AgentState)-> str:
        
        if state["error"] is not None:
            print(f"[router] error detected, routing to error node")
            return "error"
        
        intent = state['intent']
        print(f"[router] routing to: {intent}")
        
        match intent:
            case "question":
                return "answer"
            case "acknowledge":
                return "acknowledge"
            case "joke":
                return "joke"
            case _:
                print(f"[router] unknown intent: {intent}, defaulting to acknowledge")
                return "acknowledge"
            
    @staticmethod
    def route_after_answer(state: AgentState) -> str:
        if state["error"] is not None:
            print(f"[router] error after action, routing to error node")
            return "error"
        return "end"
