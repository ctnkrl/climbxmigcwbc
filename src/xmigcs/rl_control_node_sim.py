import os
import sys

import rclpy
import yaml

from xmigcs.utils.logging_utils import get_logger
from xmigcs.utils.xlog_utils import configure_xlog, xlog
from xmigcs.rl_control_node import XMIGCSControlNode


logger = get_logger(__name__)


class XMIGCSControlNode_sim(XMIGCSControlNode):
    def __init__(self):
        super().__init__()
        xlog.info("rewrite sim")
        self.robot_interface.rewrite_config({'sim': True})
        # 检查当前用户名，如果是ubuntu则抛出异常
        import getpass
        self.user_name = getpass.getuser().lower()
        if self.user_name == 'ubuntu':
            raise RuntimeError("On ubuntu user, run xmigcs")
def main(args=None):
    """主函数"""
    configure_xlog(logger_id="xmigcs_sim")
    rclpy.init(args=args)
    node = None
    try:
        node = XMIGCSControlNode_sim()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if "node" in locals() and node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
