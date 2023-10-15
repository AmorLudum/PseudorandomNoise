using Unity.Jobs;
using Unity.Collections;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

using static Unity.Mathematics.math;
using UnityEngine.Rendering.Universal;

public class HashVisualization : MonoBehaviour
{
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
    struct HashJob : IJobFor
    {
        [WriteOnly]
        public NativeArray<uint> hashes;

        public int resolution;
        public float invResolution;
        public SmallXXHash hash;
        public float3x4 domainTRS;

        public void Execute(int index)
        {
            float vf = floor(invResolution * index  + 0.00001f);
            float uf = invResolution * (index - resolution * vf + 0.5f) - 0.5f;
            vf = invResolution * (vf + 0.5f) - 0.5f; ;

            float3 p = mul(domainTRS, float4(uf, 0.0f, vf, 1.0f));
            int u = (int)floor(p.x);
            int v = (int)floor(p.z);
            int w = (int)floor(p.y);

            hashes[index] = hash.Eat(u).Eat(v).Eat(w);
        }
    }

    static int
        hashesId = Shader.PropertyToID("_Hashes"),
        configId = Shader.PropertyToID("_Config");

    [SerializeField]
    Mesh instanceMesh;

    [SerializeField]
    Material material;

    [SerializeField]
    int seed;

    [SerializeField, Range(1, 512)]
    int resolution = 16;

    [SerializeField, Range(-2f, 2f)]
    float verticalOffset = 1f;

    [SerializeField]
    SpaceTRS domain = new SpaceTRS()
    {
        scale = 8.0f
    };

    NativeArray<uint> hashes;
    ComputeBuffer hashesBuffer;

    MaterialPropertyBlock propertyBlock;

    private void OnEnable()
    {
        int length = resolution * resolution;
        hashes = new NativeArray<uint>(length, Allocator.Persistent);
        hashesBuffer = new ComputeBuffer(length, 4);

        new HashJob {
            hashes = hashes,
            resolution = resolution,
            invResolution = 1f / resolution,
            hash = SmallXXHash.Seed(seed),
            domainTRS = domain.Matrix
        }.ScheduleParallel(hashes.Length, resolution, default).Complete();

        hashesBuffer.SetData(hashes);

        propertyBlock ??= new MaterialPropertyBlock();
        propertyBlock.SetBuffer(hashesId, hashesBuffer);
        propertyBlock.SetVector(configId, new Vector4(resolution, 1f / resolution, verticalOffset / resolution));
    }

    private void OnDisable()
    {
        hashes.Dispose();
        hashesBuffer.Release();
        hashesBuffer = null;
    }

    private void OnValidate()
    {
        if (hashesBuffer != null && enabled)
        {
            OnDisable();
            OnEnable();
        }
    }

    private void Update()
    {
        Graphics.DrawMeshInstancedProcedural
        (
            instanceMesh, 0, material, new Bounds(Vector3.zero, Vector3.one),
            hashes.Length, propertyBlock
        );
    }
}