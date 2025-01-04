from metaflow import Flow
run = Flow('KMeansFlow').latest_run

k = run.data.top[4][:3]
l = run.data.top[5][:3]
m = run.data.top[6][:3]

print(f"Kluster 4: {k}")
print(f"Kluster 5: {l}")
print(f"Kluster 6: {m}")

print(k)