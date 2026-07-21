from dataclasses import dataclass

@dataclass
class FSMControlFlag:
    """状态机控制标志"""
    fsm_state_command: str = "gotoSTOP"
    fsm_resume_command: str = ""