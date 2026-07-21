#!/usr/bin/env python3
"""
Complete mapping tests for common/xbox_control.py
Tests all button mappings after removing turbo key and replacing back with select
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sensor_msgs.msg import Joy
from common.xbox_control import XBOXController


def make_joy(axes=None, buttons=None):
    j = Joy()
    j.axes = [float(x) for x in (axes or [])]
    j.buttons = [int(x) for x in (buttons or [])]
    return j


def test_gotoZERO():
    """Test X button -> gotoZERO"""
    c = XBOXController()
    # X button -> gotoZERO is buttons[3]
    j = make_joy(axes=[0]*8, buttons=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.fsm_state_command == 'gotoZERO', f"expected gotoZERO, got {c.flag.fsm_state_command}"
    print('✓ test_gotoZERO passed')


def test_gotoSTOP():
    """Test Y button -> gotoSTOP"""
    c = XBOXController()
    # Y button -> gotoSTOP is buttons[4]
    j = make_joy(axes=[0]*8, buttons=[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.fsm_state_command == 'gotoSTOP', f"expected gotoSTOP, got {c.flag.fsm_state_command}"
    print('✓ test_gotoSTOP passed')


def test_gotoDHZERO():
    """Test A button alone -> gotoDHZERO"""
    c = XBOXController()
    # A button alone -> gotoDHZERO is buttons[0]
    j = make_joy(axes=[0]*8, buttons=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.fsm_state_command == 'gotoDHZERO', f"expected gotoDHZERO, got {c.flag.fsm_state_command}"
    print('✓ test_gotoDHZERO passed')


def test_gotoPBHCZERO():
    """Test B button alone -> gotoPBHCZERO"""
    c = XBOXController()
    # B button alone -> gotoPBHCZERO is buttons[1]
    j = make_joy(axes=[0]*8, buttons=[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.fsm_state_command == 'gotoPBHCZERO', f"expected gotoPBHCZERO, got {c.flag.fsm_state_command}"
    print('✓ test_gotoPBHCZERO passed')


def test_gotoDH():
    """Test Left Trigger + A -> gotoDH"""
    c = XBOXController()
    # Left Trigger + A -> gotoDH
    j = make_joy(axes=[0,0,0,0,0,-1.0,0,0], buttons=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.fsm_state_command == 'gotoDH', f"expected gotoDH, got {c.flag.fsm_state_command}"
    print('✓ test_gotoDH passed')


def test_gotoPBHC():
    """Test Left Trigger + B -> gotoPBHC"""
    c = XBOXController()
    # Left Trigger + B -> gotoPBHC
    j = make_joy(axes=[0,0,0,0,0,-1.0,0,0], buttons=[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.fsm_state_command == 'gotoPBHC', f"expected gotoPBHC, got {c.flag.fsm_state_command}"
    print('✓ test_gotoPBHC passed')


def test_gotoMLP():
    """Test Left Trigger + X -> gotoMLP"""
    c = XBOXController()
    # Left Trigger + X -> gotoMLP
    j = make_joy(axes=[0,0,0,0,0,-1.0,0,0], buttons=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.fsm_state_command == 'gotoMLP', f"expected gotoMLP, got {c.flag.fsm_state_command}"
    print('✓ test_gotoMLP passed')


def test_gotoMLPH():
    """Test LB + X -> gotoMLPH"""
    c = XBOXController()
    # LB + X -> gotoMLPH (buttons[6] + buttons[3])
    j = make_joy(axes=[0]*8, buttons=[0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.fsm_state_command == 'gotoMLPH', f"expected gotoMLPH, got {c.flag.fsm_state_command}"
    print('✓ test_gotoMLPH passed')


def test_height_increase_select():
    """Test Select button -> increase height"""
    c = XBOXController()
    # simulate select button press -> increase height
    j = make_joy(axes=[0]*8, buttons=[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.motion_number == 1
    print('✓ test_height_increase_select passed')


def test_height_decrease_start():
    """Test Start button -> decrease height"""
    c = XBOXController()
    # simulate start button press -> decrease height
    j = make_joy(axes=[0]*8, buttons=[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.motion_number == -1
    print('✓ test_height_decrease_start passed')


def test_disable_select_b():
    """Test LB + B buttons -> disable"""
    c = XBOXController()
    # LB + B buttons -> disable (buttons[6] + buttons[1])
    j = make_joy(axes=[0]*8, buttons=[0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.disable_joints == True, f"expected is_disable=True, got {c.flag.disable_joints}"
    print('✓ test_disable_select_b passed')


def test_disable_lb_rb():
    """Test LB + RB buttons -> disable (Ctrl+C equivalent)"""
    c = XBOXController()
    # LB + RB buttons -> disable (buttons[6] + buttons[7])
    j = make_joy(axes=[0]*8, buttons=[0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.disable_joints == True, f"expected is_disable=True, got {c.flag.disable_joints}"
    print('✓ test_disable_lb_rb passed')


def test_disable_select_start():
    """Test Select + Start buttons -> disable (emergency)"""
    c = XBOXController()
    # Select + Start buttons -> disable (buttons[10] + buttons[11])
    j = make_joy(axes=[0]*8, buttons=[0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.disable_joints == True, f"expected is_disable=True, got {c.flag.disable_joints}"
    print('✓ test_disable_select_start passed')


def test_movement_controls():
    """Test movement controls with left and right sticks"""
    c = XBOXController()
    # Test with left stick up (forward movement)
    j = make_joy(axes=[0,1,0,0,0,0,0,0], buttons=[0]*16)
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.x_speed_command > 0, f"expected positive x_speed_command, got {c.flag.x_speed_command}"
    
    # Test with left stick right
    j = make_joy(axes=[1,0,0,0,0,0,0,0], buttons=[0]*16)
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.y_speed_command < 0, f"expected negative y_speed_command, got {c.flag.y_speed_command}"
    
    # Test with right stick right
    j = make_joy(axes=[0,0,1,0,0,0,0,0], buttons=[0]*16)
    c.xbox_map_read(j)
    c.xbox_flag_update()
    assert c.flag.yaw_speed_command < 0, f"expected negative yaw_speed_command, got {c.flag.yaw_speed_command}"
    print('✓ test_movement_controls passed')


def test_reset_movement():
    """Test Start button -> reset movement"""
    c = XBOXController()
    # Set some movement values first
    c.flag.x_speed_command = 0.5
    c.flag.y_speed_command = -0.3
    c.flag.yaw_speed_command = 0.2
    
    # Press start button to reset
    j = make_joy(axes=[0]*8, buttons=[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    c.xbox_map_read(j)
    c.xbox_flag_update()
    
    assert c.flag.x_speed_command == 0.0, f"expected x_speed_command=0.0, got {c.flag.x_speed_command}"
    assert c.flag.y_speed_command == 0.0, f"expected y_speed_command=0.0, got {c.flag.y_speed_command}"
    assert c.flag.yaw_speed_command == 0.0, f"expected yaw_speed_command=0.0, got {c.flag.yaw_speed_command}"
    print('✓ test_reset_movement passed')


def run_all():
    print("Running all XBOX controller mapping tests...")
    test_gotoZERO()
    test_gotoSTOP()
    test_gotoDHZERO()
    test_gotoPBHCZERO()
    test_gotoDH()
    test_gotoPBHC()
    test_gotoMLP()
    test_gotoMLPH()
    test_height_increase_select()
    test_height_decrease_start()
    test_disable_select_b()
    test_disable_lb_rb()
    test_disable_select_start()
    test_movement_controls()
    test_reset_movement()
    print("All tests passed! ✅")


if __name__ == '__main__':
    run_all()