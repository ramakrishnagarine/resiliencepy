from resiliencepy import Disruption, Policy, simulate, simulate_batch, Metrics

d = Disruption(kind="port_closure", severity=0.65, duration_days=12, start_day=2)
p = Policy(safety_stock=0.25, expediting=True, dual_sourcing=True)

series = simulate(d, p, horizon_days=80, curve_shape="logistic")
print("Single metrics:", Metrics.compute(series))

# Batch: compare strategies
policies = [
    Policy(safety_stock=0.05, expediting=False),
    Policy(safety_stock=0.25, expediting=True),
    Policy(safety_stock=0.25, expediting=True, dual_sourcing=True, rerouting=True),
]
batch = simulate_batch([d], policies, horizon_days=80)
print("Batch metrics:", Metrics.compute(batch))
