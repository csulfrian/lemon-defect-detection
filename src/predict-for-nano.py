import qwiic_relay

QUAD_RELAY = 0x6D

relay_board = qwiic_relay.QwiicRelay(QUAD_RELAY)

myRelays.set_relay_on(1)
myRelays.set_relay_on(3)
myRelays.set_all_relays_off()