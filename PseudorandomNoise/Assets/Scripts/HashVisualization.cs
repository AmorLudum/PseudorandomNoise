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
        [ReadOnly]
        public NativeArray<float3> positions;
        [WriteOnly]
        public NativeArray<uint> hashes;
        public SmallXXHash hash;
        public float3x4 domainTRS;

        public void Execute(int index)
        {
            float3 p = mul(domainTRS, float4(positions[index], 1.0f));
            int u = (int)floor(p.x);
            int v = (int)floor(p.z);
            int w = (int)floor(p.y);

            hashes[index] = hash.Eat(u).Eat(v).Eat(w);
        }
    }

    static int
        hashesId = Shader.PropertyToID("_Hashes"),
        positionsId = Shader.PropertyToID("_Positions"),
        normalsId = Shader.PropertyToID("_Normals"),
        configId = Shader.PropertyToID("_Config");

    [SerializeField]
    Mesh instanceMesh;

    [SerializeField]
    Material material;

    [SerializeField]
    int seed;

    [SerializeField, Range(1, 512)]
    int resolution = 16;

    [SerializeField, Range(-0.5f, 0.5f)]
    float displacement = 0.1f;

    [SerializeField]
    SpaceTRS domain = new SpaceTRS()
    {
        scale = 8.0f
    };

    NativeArray<uint> hashes;
    NativeArray<float3> positions, normals;
    ComputeBuffer hashesBuffer, positionsBuffer, normalsBuffer;

    MaterialPropertyBlock propertyBlock;

    bool m_dirty = true;
    Bounds m_bounds;

    private void OnEnable()
    {
        m_dirty = true;

        int length = resolution * resolution;
        hashes = new NativeArray<uint>(length, Allocator.Persistent);
        positions = new NativeArray<float3>(length, Allocator.Persistent);
        normals = new NativeArray<float3>(length, Allocator.Persistent);
        hashesBuffer = new ComputeBuffer(length, 4);
        positionsBuffer = new ComputeBuffer(length, 3 * 4);
        normalsBuffer = new ComputeBuffer(length, 3 * 4);

        propertyBlock ??= new MaterialPropertyBlock();
        propertyBlock.SetBuffer(hashesId, hashesBuffer);
        propertyBlock.SetBuffer(positionsId, positionsBuffer);
        propertyBlock.SetVector(configId, new Vector4(resolution, 1f / resolution, displacement));
        propertyBlock.SetBuffer(normalsId, normalsBuffer);
    }

    private void OnDisable()
    {
        hashes.Dispose();
        positions.Dispose();
        normals.Dispose();
        hashesBuffer.Release();
        positionsBuffer.Release();
        normalsBuffer.Release();
        hashesBuffer = null;
        positionsBuffer = null;
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
        if (m_dirty || transform.hasChanged)
        {
            m_dirty = false;
            transform.hasChanged = false;

            JobHandle handle = Shapes.Job.ScheduleParallel(positions, normals, resolution, transform.localToWorldMatrix, default);

            new HashJob
            {
                positions = positions,
                hashes = hashes,
                hash = SmallXXHash.Seed(seed),
                domainTRS = domain.Matrix
            }.ScheduleParallel(hashes.Length, resolution, handle).Complete();

            hashesBuffer.SetData(hashes);
            positionsBuffer.SetData(positions);
            normalsBuffer.SetData(normals);

            m_bounds = new Bounds
            (
                transform.position,
                float3(2f * cmax(abs(transform.lossyScale)) + displacement)
            );
        }
        Graphics.DrawMeshInstancedProcedural
        (
            instanceMesh, 0, material, m_bounds, hashes.Length, propertyBlock
        );
    }
}